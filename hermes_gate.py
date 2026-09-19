"""
Backward compatibility shim for dpai.hermes_gate
"""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dpai.hermes_gate import *
