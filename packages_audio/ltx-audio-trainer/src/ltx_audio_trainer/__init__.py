import logging
import os
import sys
from logging import getLogger
from pathlib import Path

from rich.logging import RichHandler

# Get the process rank
IS_MULTI_GPU = os.environ.get("LOCAL_RANK") is not None
RANK = int(os.environ.get("LOCAL_RANK", "0"))

# Configure with Rich
logging.basicConfig(
    level="INFO",
    format=f"\\[rank {RANK}] %(message)s" if IS_MULTI_GPU else "%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            markup=True,
        )
    ],
)

# Get the logger and configure it
logger = getLogger("ltx_audio_trainer")
logger.setLevel(logging.DEBUG)
logger.propagate = True

# Set level based on process
if RANK != 0:
    logger.setLevel(logging.WARNING)

# Expose common logging functions directly
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

package_root = Path(__file__).parent.parent.parent
workspace_audio_core_src = package_root.parent / "ltx-audio-core" / "src"

# Add local workspace roots so scripts/tests can import sibling packages without installation.
sys.path.insert(0, str(package_root))
if workspace_audio_core_src.exists():
    sys.path.insert(0, str(workspace_audio_core_src))
