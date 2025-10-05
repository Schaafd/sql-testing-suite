from __future__ import annotations

from pathlib import Path
import shutil

import pytest


@pytest.fixture
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide a temporary directory compatible with legacy fixtures."""
    path = tmp_path_factory.mktemp("sqltest")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
