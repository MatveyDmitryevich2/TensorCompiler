# TensorCompiler

tensor compiler for sber compiler class

## Prerequisites

- CMake ≥ 3.21
- protoc
- Graphviz (optional)
- Python 3
- mlir-opt
- mlir-translate
- llc
- C++ compiler available as `c++` or `CXX`
- C++20
- Python packages from the test scripts: `onnx`, `numpy`

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Generate model

```bash
python3 tests/main_ops.py
```

This writes `run_data/models/main_ops.onnx`.

## Usage

```bash
./build/tc.x <model_path> [options]
```

### Options

```text
--emit-dot <path>
--emit-mlir <path>
--emit-llvm <path>
--emit-asm <path>
--target-triple <triple>
--mcpu <cpu>
--O0 | --O1 | --O2 | --O3
--run
--run-compiled
--input <name=path>
--output-dir <dir>
```

## Examples

```bash
mkdir -p run_data/artifacts

./build/tc.x run_data/models/main_ops.onnx --emit-dot run_data/artifacts/main_ops.dot
./build/tc.x run_data/models/main_ops.onnx --emit-mlir run_data/artifacts/main_ops.mlir
./build/tc.x run_data/models/main_ops.onnx --emit-llvm run_data/artifacts/main_ops.ll
./build/tc.x run_data/models/main_ops.onnx --emit-asm run_data/artifacts/main_ops.s
./build/tc.x run_data/models/main_ops.onnx --emit-asm run_data/artifacts/main_ops.s --target-triple x86_64-pc-linux-gnu --mcpu native --O3
```

## Execute model

The built-in CPU runtime supports float32 tensors and the project operations:
`Add`, `Mul`, `Conv`, `Relu`, `MatMul`, `Gemm`, `Transpose`.

Input files are whitespace/comma separated float values in row-major order.
Keep local inputs and outputs under `run_data/`; this directory is ignored by git.

```bash
mkdir -p run_data/inputs run_data/outputs

python3 - <<'PY'
from pathlib import Path
Path("run_data/inputs/X.txt").write_text(" ".join(["1.0"] * (1 * 3 * 8 * 8)))
Path("run_data/inputs/A.txt").write_text(" ".join(["1.0"] * (2 * 3)))
PY

./build/tc.x run_data/models/main_ops.onnx \
  --run \
  --input X=run_data/inputs/X.txt \
  --input A=run_data/inputs/A.txt \
  --output-dir run_data/outputs
```

Runtime logs are written to `run_data/logs/tc.log`.

To execute the compiled LLVM/native path instead of the built-in interpreter:

```bash
./build/tc.x run_data/models/main_ops.onnx \
  --run-compiled \
  --input X=run_data/inputs/X.txt \
  --input A=run_data/inputs/A.txt \
  --output-dir run_data/compiled_outputs
```

`--run-compiled` lowers MLIR to LLVM IR, compiles it to a native object with `llc`,
links a small generated C++ runner, and executes the resulting binary.

## Compare with ONNX reference

```bash
python3 tests/compare_runtime.py
```

This checks both `--run` and `--run-compiled` against ONNX ReferenceEvaluator.

## Generate graph img

```bash
bash dot2svg.sh run_data/artifacts/main_ops.dot
```
