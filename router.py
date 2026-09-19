"""
Backward compatibility shim for dpai.router
"""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dpai.router import *
from dpai.router import main

if __name__ == "__main__":
    main()
