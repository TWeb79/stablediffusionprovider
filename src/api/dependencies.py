"""
Shared dependencies for API routes.

Author: Inventions4All - github:TWeb79
"""

from functools import lru_cache

from ..core.config import Settings, get_settings
from ..core.pipeline import PipelineManager, get_pipeline_manager


@lru_cache
def get_settings_dep() -> Settings:
    """Dependency for getting settings."""
    return get_settings()


def get_pipeline_dep() -> PipelineManager:
    """Dependency for getting pipeline manager."""
    return get_pipeline_manager()