from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from main_ops import build_demo_onnx


def write_values(path: Path, values: np.ndarray) -> None:
    path.write_text(" ".join(str(float(x)) for x in values.reshape(-1)))


def read_values(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromstring(path.read_text(), sep=" ", dtype=np.float32)
    return values.reshape(shape)


def run_tc(tc: Path,
           model_path: Path,
           x_path: Path,
           a_path: Path,
           out_dir: Path,
           compiled: bool) -> None:
    env = os.environ.copy()
    env.setdefault("LSAN_OPTIONS", "detect_leaks=0")
    subprocess.run(
        [
            str(tc),
            str(model_path),
            "--run-compiled" if compiled else "--run",
            "--input",
            f"X={x_path}",
            "--input",
            f"A={a_path}",
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        env=env,
    )


def check_outputs(out_dir: Path, expected: dict[str, np.ndarray], label: str) -> float:
    max_diff = 0.0
    for name, exp in expected.items():
        got = read_values(out_dir / f"{name}.txt", exp.shape)
        diff = float(np.max(np.abs(got - exp)))
        print(f"{label}/{name}: max_abs_diff={diff:.6g}")
        max_diff = max(max_diff, diff)
    return max_diff


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "build"
    tc = build_dir / "tc.x"
    work_dir = build_dir / "runtime_compare"
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = work_dir / "main_ops.onnx"
    build_demo_onnx(str(model_path))

    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 3, 8, 8), dtype=np.float32)
    a = rng.standard_normal((2, 3), dtype=np.float32)

    x_path = work_dir / "X.txt"
    a_path = work_dir / "A.txt"
    write_values(x_path, x)
    write_values(a_path, a)

    model = onnx.load(str(model_path))
    ref = ReferenceEvaluator(model)
    expected = dict(zip(["Y_gemm", "X_t"], ref.run(None, {"X": x, "A": a})))

    runtime_out = work_dir / "runtime_outputs"
    compiled_out = work_dir / "compiled_outputs"
    run_tc(tc, model_path, x_path, a_path, runtime_out, compiled=False)
    run_tc(tc, model_path, x_path, a_path, compiled_out, compiled=True)

    max_diff = max(
        check_outputs(runtime_out, expected, "runtime"),
        check_outputs(compiled_out, expected, "compiled"),
    )

    if max_diff > 1e-4:
        print(f"runtime mismatch: max_abs_diff={max_diff:.6g}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
