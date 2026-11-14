"""Draft persistence and management."""

import json
import uuid
from typing import List

from app.core.draft_models import Draft
from core.config import PROJECT_ROOT
from core.logging import logger

DRAFTS_DIR = PROJECT_ROOT / "app" / "drafts"


def new_draft_id() -> str:
    """Generate a new short draft ID."""
    return str(uuid.uuid4())[:8]


def save_draft(draft: Draft) -> None:
    """Save a draft to JSON file.

    Args:
        draft: Draft object to save

    Raises:
        Exception: If saving fails
    """
    drafts_dir = DRAFTS_DIR
    drafts_dir.mkdir(exist_ok=True)

    draft_file = drafts_dir / f"{draft.id}.json"

    try:
        # Convert to dict for JSON serialization
        draft_dict = draft.model_dump()

        # Convert datetime to ISO format
        draft_dict["created_at"] = draft.created_at.isoformat()

        with open(draft_file, "w", encoding="utf-8") as f:
            json.dump(draft_dict, f, indent=2, ensure_ascii=False)

        logger.debug(f"Draft {draft.id} saved to {draft_file}")

    except Exception as e:
        logger.error(f"Failed to save draft {draft.id}: {e}")
        raise Exception(f"Failed to save draft: {e}")


def load_draft(draft_id: str) -> Draft:
    """Load a draft from JSON file.

    Args:
        draft_id: ID of the draft to load

    Returns:
        Draft object

    Raises:
        FileNotFoundError: If draft doesn't exist
        Exception: If loading/parsing fails
    """
    drafts_dir = DRAFTS_DIR
    draft_file = drafts_dir / f"{draft_id}.json"

    if not draft_file.exists():
        raise FileNotFoundError(f"Draft {draft_id} not found")

    try:
        with open(draft_file, encoding="utf-8") as f:
            draft_dict = json.load(f)

        # Parse back into Draft object
        draft = Draft(**draft_dict)

        logger.debug(f"Draft {draft_id} loaded successfully")
        return draft

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in draft {draft_id}: {e}")
        raise Exception(f"Invalid draft format: {e}")
    except Exception as e:
        logger.error(f"Failed to load draft {draft_id}: {e}")
        raise Exception(f"Failed to load draft: {e}")


def list_drafts() -> List[str]:
    """List all available draft IDs.

    Returns:
        List of draft IDs (filenames without .json extension)
    """
    drafts_dir = DRAFTS_DIR

    if not drafts_dir.exists():
        return []

    try:
        draft_files = drafts_dir.glob("*.json")
        # Skip sidecar files such as <id>.proposals.json (pending refinements).
        draft_ids = [f.stem for f in draft_files if f.is_file() and "." not in f.stem]
        draft_ids.sort()  # Sort for consistent ordering

        logger.debug(f"Found {len(draft_ids)} drafts")
        return draft_ids

    except Exception as e:
        logger.error(f"Failed to list drafts: {e}")
        return []
