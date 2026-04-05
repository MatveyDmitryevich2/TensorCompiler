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
- C++20

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
```

## Examples

```bash
./build/tc.x main_ops.onnx --emit-dot out.dot
./build/tc.x main_ops.onnx --emit-mlir out.mlir
./build/tc.x main_ops.onnx --emit-llvm out.ll
./build/tc.x main_ops.onnx --emit-asm out.s
./build/tc.x main_ops.onnx --emit-asm out.s --target-triple x86_64-pc-linux-gnu --mcpu native --O3
```

## Generate graph img

```bash
bash dot2svg.sh out.dot
```