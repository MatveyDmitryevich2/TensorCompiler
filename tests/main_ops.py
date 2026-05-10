from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference


def make_tensor(name, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr.astype(np.float32), name=name)


def build_demo_onnx(path: str = "tc_demo.onnx", opset: int = 19) -> onnx.ModelProto:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Conv branch input (NCHW)
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])

    # MatMul/Add/Mul/Gemm branch inputs (2D)
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])

    rng = np.random.default_rng(0)

    # Grouped Conv: X has 3 channels, group=3 means each group consumes 1 channel.
    w_conv = make_tensor("W_conv", rng.standard_normal(
        (6, 1, 3, 3), dtype=np.float32))
    b_conv = make_tensor("B_conv", rng.standard_normal((6,), dtype=np.float32))

    b_matmul = make_tensor(
        "B_matmul", rng.standard_normal((3, 4), dtype=np.float32))
    # Add broadcasts this row vector over the leading dimension.
    c_add = make_tensor("C_add", rng.standard_normal((4,), dtype=np.float32))
    # Mul broadcasts this scalar over the whole matrix.
    s_mul = make_tensor("S_mul", np.array(0.5, dtype=np.float32))

    # Gemm uses both transA and transB, plus a broadcast bias vector.
    b_gemm = make_tensor("B_gemm", rng.standard_normal(
        (5, 4), dtype=np.float32))  # used with transB=1
    c_gemm = make_tensor("C_gemm", rng.standard_normal((5,), dtype=np.float32))

    initializers = [w_conv, b_conv, b_matmul, c_add, s_mul, b_gemm, c_gemm]

    nodes = []

    # Conv -> Relu -> Transpose with explicit perm.
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["X", "W_conv", "B_conv"],
            outputs=["X_conv"],
            name="conv0",
            strides=[1, 1],
            pads=[1, 1, 1, 1],       # top, left, bottom, right
            dilations=[1, 1],
            group=3,
        )
    )
    nodes.append(
        helper.make_node(
            "Relu",
            inputs=["X_conv"],
            outputs=["X_relu"],
            name="relu0",
        )
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            inputs=["X_relu"],
            outputs=["X_t"],
            name="transpose0",
            perm=[0, 2, 3, 1],       # NCHW -> NHWC
        )
    )

    # MatMul -> broadcast Add -> scalar Mul -> default Transpose -> Gemm.
    nodes.append(
        helper.make_node(
            "MatMul",
            inputs=["A", "B_matmul"],
            outputs=["Y_mm"],
            name="matmul0",
        )
    )
    nodes.append(
        helper.make_node(
            "Add",
            inputs=["Y_mm", "C_add"],
            outputs=["Y_add"],
            name="add0",
        )
    )
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=["Y_add", "S_mul"],
            outputs=["Y_mul"],
            name="mul0",
        )
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            inputs=["Y_mul"],
            outputs=["Y_mul_t"],
            name="transpose_default0",
        )
    )
    # Gemm: Y = alpha * A' * B' + beta * C
    # A = Y_mul_t [4x2] with transA=1 => [2x4].
    # B = B_gemm [5x4] with transB=1 => [4x5].
    # C = C_gemm [5] broadcasts over rows, output [2x5].
    nodes.append(
        helper.make_node(
            "Gemm",
            inputs=["Y_mul_t", "B_gemm", "C_gemm"],
            outputs=["Y_gemm"],
            name="gemm0",
            alpha=1.2,
            beta=0.7,
            transA=1,
            transB=1,
        )
    )

    y_gemm = helper.make_tensor_value_info("Y_gemm", TensorProto.FLOAT, [2, 5])
    x_t = helper.make_tensor_value_info("X_t", TensorProto.FLOAT, [1, 8, 8, 6])

    graph = helper.make_graph(
        nodes=nodes,
        name="tc_demo_graph",
        inputs=[x, a],
        outputs=[y_gemm, x_t],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="tc-demo",
        opset_imports=[helper.make_opsetid("", opset)],
    )

    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    onnx.save(model, path)
    return model


if __name__ == "__main__":
    build_demo_onnx("run_data/models/main_ops.onnx", opset=19)
