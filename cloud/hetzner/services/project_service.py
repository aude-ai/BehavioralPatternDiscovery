"""Project service for business logic."""
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from ..database import ProjectModel, JobModel
from ..models import (
    Project, ProjectCreate, ProjectUpdate, ProjectStatus,
    Job, JobStatus, JobType,
    VariantCreate, BorrowedFile,
)
from .storage_service import StorageService
from .r2_service import delete_all_project_r2_files

logger = logging.getLogger(__name__)

# All trackable file keys for variant ownership
ALL_FILE_KEYS = [
    # R2 files
    "embeddings",
    "aux_features",
    "normalized_embeddings",
    "train_input",
    "message_database",
    "checkpoint",
    "activations",
    # Hetzner files
    "activities",
    "population_stats",
    "message_examples",
    "hierarchical_weights",
    "pattern_names",
]

# Default naming for steps
STEP_NAMES = {
    "B.1": "Statistical Features",
    "B.2": "Text Embedding",
    "B.3": "Normalization",
    "B.4": "Training Preparation",
    "B.5": "Training",
    "B.6": "Batch Scoring",
    "B.7": "Message Assignment",
    "B.8": "SHAP Analysis",
}


class ProjectService:
    """Service for project operations."""

    def __init__(self, db: Session):
        self.db = db

    def create_project(self, data: ProjectCreate) -> Project:
        """Create a new project."""
        project_id = str(uuid4())

        # Create database record
        db_project = ProjectModel(
            id=project_id,
            name=data.name,
            description=data.description,
            config_overrides=data.config_overrides,
            status=ProjectStatus.CREATED,
        )
        self.db.add(db_project)
        self.db.commit()
        self.db.refresh(db_project)

        # Create storage directories
        storage = StorageService(project_id)
        storage.ensure_directories()

        return Project.model_validate(db_project)

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        db_project = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not db_project:
            return None
        return Project.model_validate(db_project)

    def list_projects(self, skip: int = 0, limit: int = 100) -> list[Project]:
        """List all projects."""
        db_projects = self.db.query(ProjectModel).offset(skip).limit(limit).all()
        return [Project.model_validate(p) for p in db_projects]

    def update_project(self, project_id: str, data: ProjectUpdate) -> Optional[Project]:
        """Update project."""
        db_project = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not db_project:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_project, key, value)

        self.db.commit()
        self.db.refresh(db_project)
        return Project.model_validate(db_project)

    def update_status(self, project_id: str, status: ProjectStatus):
        """Update project status."""
        self.db.query(ProjectModel).filter(ProjectModel.id == project_id).update(
            {"status": status, "updated_at": datetime.utcnow()}
        )
        self.db.commit()

    def delete_project(self, project_id: str) -> dict:
        """Delete project and all associated data.

        Returns:
            dict with either:
            - {"deleted": True, "detached_variants": [...]} on success
            - {"error": "deletion_blocked", "message": ..., "borrowed_files": [...]} if blocked
        """
        db_project = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not db_project:
            return {"error": "not_found", "message": "Project not found"}

        logger.info(f"Attempting to delete project {project_id}")

        # Check for variants that borrow files from this project
        variants = self.get_variants(project_id)
        borrowed_files = []

        for variant in variants:
            owned = variant.owned_files or {}
            for file_key in ALL_FILE_KEYS:
                if not owned.get(file_key, False):
                    borrowed_files.append(BorrowedFile(
                        variant_id=variant.id,
                        variant_name=variant.name,
                        file_key=file_key,
                    ))

        if borrowed_files:
            logger.warning(f"Cannot delete project {project_id}: {len(borrowed_files)} files still borrowed by variants")
            return {
                "error": "deletion_blocked",
                "message": "Cannot delete: variants still borrow files from this project",
                "borrowed_files": [bf.model_dump() for bf in borrowed_files],
            }

        # All variants own all their files - safe to delete
        # Detach variants (they become root projects)
        detached_variant_ids = []
        for variant in variants:
            self.db.query(ProjectModel).filter(ProjectModel.id == variant.id).update(
                {"parent_id": None, "updated_at": datetime.utcnow()}
            )
            detached_variant_ids.append(variant.id)
            logger.info(f"Detached variant {variant.id} from parent {project_id}")

        # Delete R2 files (only files this project owns)
        owned_files = db_project.owned_files or {key: True for key in ALL_FILE_KEYS}
        r2_failed = []
        for file_key in ALL_FILE_KEYS:
            if owned_files.get(file_key, True):
                from .r2_service import delete_r2_file
                if not delete_r2_file(project_id, file_key):
                    r2_failed.append(file_key)

        if r2_failed:
            logger.warning(f"Some R2 files failed to delete: {r2_failed}")

        # Delete Hetzner storage
        storage = StorageService(project_id)
        storage.delete_project()
        logger.info(f"Deleted Hetzner storage for project {project_id}")

        # Delete jobs
        job_count = self.db.query(JobModel).filter(JobModel.project_id == project_id).delete()
        logger.info(f"Deleted {job_count} jobs for project {project_id}")

        # Delete project
        self.db.delete(db_project)
        self.db.commit()
        logger.info(f"Project {project_id} deleted successfully")

        return {"deleted": True, "detached_variants": detached_variant_ids}

    # Job operations
    def create_job(self, project_id: str, job_type: JobType) -> Job:
        """Create a new job."""
        db_job = JobModel(
            id=str(uuid4()),
            project_id=project_id,
            job_type=job_type,
            status=JobStatus.PENDING,
        )
        self.db.add(db_job)
        self.db.commit()
        self.db.refresh(db_job)
        return Job.model_validate(db_job)

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        db_job = self.db.query(JobModel).filter(JobModel.id == job_id).first()
        if not db_job:
            return None
        return Job.model_validate(db_job)

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        progress_message: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None,
        modal_call_id: Optional[str] = None,
    ):
        """Update job fields."""
        update_data = {}
        if status is not None:
            update_data["status"] = status
            if status == JobStatus.RUNNING:
                update_data["started_at"] = datetime.utcnow()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                update_data["completed_at"] = datetime.utcnow()
        if progress is not None:
            update_data["progress"] = progress
        if progress_message is not None:
            update_data["progress_message"] = progress_message
        if result is not None:
            update_data["result"] = result
        if error is not None:
            update_data["error"] = error
        if modal_call_id is not None:
            update_data["modal_call_id"] = modal_call_id

        if update_data:
            self.db.query(JobModel).filter(JobModel.id == job_id).update(update_data)
            self.db.commit()

    def get_project_jobs(
        self,
        project_id: str,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
    ) -> list[Job]:
        """Get jobs for a project."""
        query = self.db.query(JobModel).filter(JobModel.project_id == project_id)
        if job_type:
            query = query.filter(JobModel.job_type == job_type)
        if status:
            query = query.filter(JobModel.status == status)
        return [Job.model_validate(j) for j in query.all()]

    # Variant operations (Phase 5)

    def create_variant(self, parent_id: str, data: VariantCreate) -> dict:
        """Create a variant of an existing project.

        Returns:
            dict with either the new Project or an error.
        """
        parent = self.db.query(ProjectModel).filter(ProjectModel.id == parent_id).first()
        if not parent:
            return {"error": "not_found", "message": "Parent project not found"}

        # Check parent is a root project (not a variant)
        if parent.parent_id is not None:
            return {"error": "invalid_parent", "message": "Cannot create variant of a variant"}

        # Generate name if not provided
        name = data.name
        if not name:
            step_name = STEP_NAMES.get(data.starting_step, data.starting_step)
            name = f"{parent.name} (from {step_name})"

        # Create variant with all files as inherited (owned_files all False)
        variant_id = str(uuid4())
        owned_files = {key: False for key in ALL_FILE_KEYS}

        db_variant = ProjectModel(
            id=variant_id,
            name=name,
            description=parent.description,
            config_overrides=parent.config_overrides,
            status=ProjectStatus.CREATED,
            parent_id=parent_id,
            owned_files=owned_files,
        )
        self.db.add(db_variant)
        self.db.commit()
        self.db.refresh(db_variant)

        # Create storage directories for variant
        storage = StorageService(variant_id)
        storage.ensure_directories()

        logger.info(f"Created variant {variant_id} of {parent_id}, starting from {data.starting_step}")
        return {"variant": Project.model_validate(db_variant)}

    def get_variants(self, parent_id: str) -> list[Project]:
        """Get all variants of a project."""
        db_variants = self.db.query(ProjectModel).filter(ProjectModel.parent_id == parent_id).all()
        return [Project.model_validate(v) for v in db_variants]

    def get_variant_count(self, parent_id: str) -> int:
        """Get the number of variants for a project."""
        return self.db.query(ProjectModel).filter(ProjectModel.parent_id == parent_id).count()

    def update_owned_files(self, project_id: str, file_key: str, owned: bool = True):
        """Mark a file as owned (or not owned) by a project."""
        db_project = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not db_project:
            return

        owned_files = db_project.owned_files or {}
        owned_files[file_key] = owned
        db_project.owned_files = owned_files
        db_project.updated_at = datetime.utcnow()
        self.db.commit()
        logger.info(f"Updated owned_files for {project_id}: {file_key}={owned}")

    def set_multiple_owned_files(self, project_id: str, file_keys: list[str], owned: bool = True):
        """Mark multiple files as owned (or not owned) by a project."""
        db_project = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not db_project:
            return

        owned_files = db_project.owned_files or {}
        for file_key in file_keys:
            owned_files[file_key] = owned
        db_project.owned_files = owned_files
        db_project.updated_at = datetime.utcnow()
        self.db.commit()
        logger.info(f"Updated owned_files for {project_id}: {file_keys}={owned}")

    def delete_variant(self, variant_id: str) -> bool:
        """Delete a variant project. Only deletes files the variant owns."""
        db_variant = self.db.query(ProjectModel).filter(ProjectModel.id == variant_id).first()
        if not db_variant:
            return False

        # Verify this is actually a variant
        if db_variant.parent_id is None:
            logger.error(f"Attempted to delete non-variant {variant_id} via delete_variant")
            return False

        logger.info(f"Deleting variant {variant_id}")

        # Only delete files this variant owns
        owned_files = db_variant.owned_files or {}
        from .r2_service import delete_r2_file

        for file_key in ALL_FILE_KEYS:
            if owned_files.get(file_key, False):
                delete_r2_file(variant_id, file_key)
                logger.info(f"Deleted owned file {file_key} for variant {variant_id}")

        # Delete Hetzner storage (only owned files exist there)
        storage = StorageService(variant_id)
        storage.delete_project()

        # Delete jobs
        job_count = self.db.query(JobModel).filter(JobModel.project_id == variant_id).delete()
        logger.info(f"Deleted {job_count} jobs for variant {variant_id}")

        # Delete variant record
        self.db.delete(db_variant)
        self.db.commit()
        logger.info(f"Variant {variant_id} deleted successfully")

        return True
