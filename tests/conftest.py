"""CPU contract tests use real FastAPI and a stubbed accelerator only."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = MagicMock()
torch.cuda.is_available.return_value = False
torch.version.hip = None
torch.version.cuda = None
sys.modules.setdefault("torch", torch)
