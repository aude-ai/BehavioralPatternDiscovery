#!/usr/bin/env python3
"""Migration script to add Phase 5 variant columns to existing database."""
import sqlite3
import sys
from pathlib import Path


def migrate(db_path: str):
    """Add parent_id and owned_files columns to projects table."""
    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(projects)")
    columns = {row[1] for row in cursor.fetchall()}

    migrations_needed = []

    if "parent_id" not in columns:
        migrations_needed.append(
            "ALTER TABLE projects ADD COLUMN parent_id VARCHAR(36) NULL"
        )

    if "owned_files" not in columns:
        migrations_needed.append(
            "ALTER TABLE projects ADD COLUMN owned_files JSON DEFAULT '{}'"
        )

    if not migrations_needed:
        print("Database already up to date - no migration needed")
        conn.close()
        return

    print(f"Running {len(migrations_needed)} migrations...")

    for sql in migrations_needed:
        print(f"  Executing: {sql}")
        cursor.execute(sql)

    # Create index for parent_id if it doesn't exist
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='index' AND name='ix_projects_parent_id'
    """)
    if not cursor.fetchone():
        print("  Creating index on parent_id...")
        cursor.execute("CREATE INDEX ix_projects_parent_id ON projects(parent_id)")

    conn.commit()
    conn.close()

    print("Migration complete!")


if __name__ == "__main__":
    # Default path for Docker container
    default_path = "/app/data/bpd.db"

    # Also check common local paths
    local_paths = [
        "data/bpd.db",
        "bpd.db",
        "./bpd.db",
    ]

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    elif Path(default_path).exists():
        db_path = default_path
    else:
        for p in local_paths:
            if Path(p).exists():
                db_path = p
                break
        else:
            print("Error: Could not find database file")
            print(f"Usage: {sys.argv[0]} <path_to_bpd.db>")
            sys.exit(1)

    migrate(db_path)
