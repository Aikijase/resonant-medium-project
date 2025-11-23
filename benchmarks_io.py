# ~/resonant-medium-project/benchmarks_io.py
import json
from pathlib import Path
BENCH_JSON = Path.home()/ "resonant-medium-project"/"data"/"benchmarks"/"benchmarks_fit_summary.json"
def load_benchmarks(path: Path = BENCH_JSON):
    return json.load(open(path))
# benchmarks_io.py
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass

BENCH_DIR = Path.home() / "resonant-medium-project" / "data" / "benchmarks"
BENCH_JSON = BENCH_DIR / "benchmarks_fit_summary.json"

@dataclass
class ABHFits:
    E_y0: float; E_A: float; E_k: float
    Q_y0: float; Q_A: float; Q_k: float

@dataclass
class PlasmaFits:
    L_A: float; L_B: float       # ω_L ≈ 1 + A(1 - e^{-k/B})
    T_y0: float; T_A: float; T_k: float  # ω_T ≈ y0 + A e^{-k·k}

@dataclass
class Benchmarks:
    abh: ABHFits
    plasma: PlasmaFits

def load_benchmarks(path: Path = BENCH_JSON) -> Benchmarks:
    J = json.load(open(path))
    # ABH energy fit
    Eb = J["ABH"]["E_fit"]
    Qb = J["ABH"]["Q_fit"]
    PbL = J["DustyPlasma"]["L_fit"]
    PbT = J["DustyPlasma"]["T_fit"]
    return Benchmarks(
        abh=ABHFits(
            E_y0=float(Eb["y0"]), E_A=float(Eb["A"]), E_k=float(Eb["k"]),
            Q_y0=float(Qb["y0"]), Q_A=float(Qb["A"]), Q_k=float(Qb["k"]),
        ),
        plasma=PlasmaFits(
            L_A=float(PbL["A"]), L_B=float(PbL["B"]),
            T_y0=float(PbT["y0"]), T_A=float(PbT["A"]), T_k=float(PbT["k"]),
        )
    )
