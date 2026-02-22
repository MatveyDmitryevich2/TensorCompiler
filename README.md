# TensorCompiler

tensor compiler for sber compiler class

## Prerequisites

- CMake ≥ 3.21
- protoc
- Graphviz (optional)
- A C++20 compiler

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```bash
./build/tc.x <model_path> [<dot_path>]
```