from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

@dataclass(frozen=True)
class OHLC:
    open: float
    high: float
    low: float
    close: float
