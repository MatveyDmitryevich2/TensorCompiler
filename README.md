# TensorCompiler

TensorCompiler - учебный компилятор нейросетевых графов. Проект загружает ONNX-модель, строит внутреннее представление графа, умеет выполнять его через встроенный CPU runtime, а также генерировать MLIR, LLVM IR и assembly для AOT-запуска.

Поддерживаемые операции в текущей версии: `Add`, `Mul`, `Conv`, `Relu`, `MatMul`, `Gemm`, `Transpose`.

## Что используется

- C++20
- CMake
- GoogleTest
- ONNX / protobuf
- MLIR tooling: `mlir-opt`, `mlir-translate`
- LLVM `llc`
- Graphviz
- Python 3
- Python: `onnx`, `numpy`

## Требования

```bash
cmake
protoc
c++
python3
mlir-opt
mlir-translate
llc
dot
```

## Сборка

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j16
```

## Тесты

```bash
ctest --test-dir build --output-on-failure
```

## Генерация demo ONNX-модели

```bash
python3 tests/main_ops.py
```

Создасться модель:

```bash
run_data/models/main_ops.onnx
```

Модель покрывает основные возможности проекта: `Conv`, `Relu`, `Transpose`, `MatMul`, `Add`, `Mul`, `Gemm`, broadcasting, grouped convolution, `transA`/`transB` у `Gemm`.

## Подготовка входных данных

Для `run_data/models/main_ops.onnx` нужны два входа:

- `X` с shape `[1,3,8,8]`, всего 192 числа `float32`
- `A` с shape `[2,3]`, всего 6 чисел `float32`

Файлы входов - обычный текст: числа через пробелы или запятые, в row-major порядке.

```bash
mkdir -p run_data/inputs run_data/outputs

python3 - <<'PY'
from pathlib import Path
Path("run_data/inputs/X.txt").write_text(" ".join(["1.0"] * (1 * 3 * 8 * 8)))
Path("run_data/inputs/A.txt").write_text(" ".join(["1.0"] * (2 * 3)))
PY
```

## Обычный запуск через встроенный runtime

```bash
./build/tc.x run_data/models/main_ops.onnx \
  --run \
  --input X=run_data/inputs/X.txt \
  --input A=run_data/inputs/A.txt \
  --output-dir run_data/outputs
```

- загружает `run_data/models/main_ops.onnx`
- выполняет граф встроенным C++ CPU runtime
- читает вход `X` из `run_data/inputs/X.txt`
- читает вход `A` из `run_data/inputs/A.txt`
- сохраняет выходы в `run_data/outputs`

После запуска появятся файлы:

```bash
run_data/outputs/X_t.txt
run_data/outputs/Y_gemm.txt
```

## AOT/compiled запуск

```bash
./build/tc.x run_data/models/main_ops.onnx \
  --run-compiled \
  --input X=run_data/inputs/X.txt \
  --input A=run_data/inputs/A.txt \
  --output-dir run_data/compiled_outputs
```

`--run-compiled` строит MLIR, опускает его до LLVM IR, компилирует object через `llc`, собирает временный C++ runner и запускает получившийся бинарник.

## Генерация артефактов

Создать директорию под результаты:

```bash
mkdir -p run_data/artifacts
```

Сгенерировать DOT-граф:

```bash
./build/tc.x run_data/models/main_ops.onnx \
  --emit-dot run_data/artifacts/main_ops.dot
```

Сгенерировать SVG из DOT:

```bash
bash dot2svg.sh run_data/artifacts/main_ops.dot
```

Результат:

```bash
run_data/artifacts/main_ops.dot.svg
```

Сгенерировать MLIR:

```bash
./build/tc.x run_data/models/main_ops.onnx \
  --emit-mlir run_data/artifacts/main_ops.mlir
```

Сгенерировать LLVM IR:

```bash
./build/tc.x run_data/models/main_ops.onnx \
  --emit-llvm run_data/artifacts/main_ops.ll
```

Сгенерировать assembly:

```bash
./build/tc.x run_data/models/main_ops.onnx \
  --emit-asm run_data/artifacts/main_ops.s
```

Сгенерировать assembly с настройками LLVM:

```bash
./build/tc.x run_data/models/main_ops.onnx \
  --emit-asm run_data/artifacts/main_ops_O3.s \
  --target-triple x86_64-pc-linux-gnu \
  --mcpu native \
  --O3
```

## Все опции `tc.x`

Общий формат:

```bash
./build/tc.x <model_path> [options]
```

Опции:

```text
--run
```

Запустить модель через встроенный CPU runtime.

```text
--run-compiled
```

Скомпилировать модель через MLIR/LLVM и запустить native-код.

```text
--input <name=path>
```

```text
--output-dir <dir>
```

Директория, куда runtime или compiled runner сохраняет выходы модели в `.txt`.

```text
--emit-dot <path>
```

Сохранить внутренний граф в DOT-формате для Graphviz.

```text
--emit-mlir <path>
```

Сохранить сгенерированный MLIR.

```text
--emit-llvm <path>
```

Опустить MLIR до LLVM IR и сохранить `.ll`.

```text
--emit-asm <path>
```

Опустить MLIR до LLVM IR, затем через `llc` сгенерировать assembly.

```text
--target-triple <triple>
```

Передать LLVM target triple для `llc`.

## Сравнение с ONNX reference

```bash
python3 tests/compare_runtime.py
```

Скрипт генерирует demo-модель, создает входы, запускает оба режима (`--run` и `--run-compiled`) и сравнивает результаты с ONNX `ReferenceEvaluator`.

## Логи

Логи запуска пишутся сюда:

```bash
run_data/logs/tc.log
```