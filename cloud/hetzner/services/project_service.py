"""Project service for business logic."""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from ..database import ProjectModel, JobModel
from ..models import Project, ProjectCreate, ProjectUpdate, ProjectStatus, Job, JobStatus, JobType
from .storage_service import StorageService


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

    def delete_project(self, project_id: str) -> bool:
        """Delete project and all associated data."""
        db_project = self.db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not db_project:
            return False

        # Delete storage
        storage = StorageService(project_id)
        storage.delete_project()

        # Delete jobs
        self.db.query(JobModel).filter(JobModel.project_id == project_id).delete()

        # Delete project
        self.db.delete(db_project)
        self.db.commit()
        return True

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
