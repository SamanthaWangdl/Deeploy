#!/usr/bin/env python3
"""Generate 4 single-kernel ConvGrad tests (DW/PW × X/W) that mirror
MobileNet-block shapes, so we can pinpoint which of the 4 MobileNet-specific
kernels produces the ~1.7% loss drift.

Creates:
  DeeployTest/Tests/Kernels/FP32/ConvGradX_DW
  DeeployTest/Tests/Kernels/FP32/ConvGradW_DW
  DeeployTest/Tests/Kernels/FP32/ConvGradX_PW
  DeeployTest/Tests/Kernels/FP32/ConvGradW_PW

Each directory gets:
  network.onnx  — single ConvGradX or ConvGradW node
  inputs.npz    — random dY/X/W (seeded, reproducible)
  outputs.npz   — PyTorch-computed dX or dW reference

Reference kernel is computed via torch.nn.grad.{conv2d_input, conv2d_weight},
which is the same definition PyTorch uses internally for backward(); this is
the ground truth the Deeploy kernel must match bit-ish-exact.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper, save_model


# ─────────────────────────────────────────────────────────────────────────────
# Shape choices — one representative per kernel path
# ─────────────────────────────────────────────────────────────────────────────
# MobileNetV1 0.25x for VWW has blocks with channels 8,16,32,64,128,256 (the
# last block goes 512→1024 in the full arch but the tiny variant stops at
# 256). Picking mid-depth block 5-ish shapes that fit in L2 easily:
#   DW block with C=64, H=W=12, stride=1, pad=1
#   PW block C_in=64 → C_out=128, H=W=12
# These are intentionally small so gvsoc finishes quickly (~minutes).

# DW ConvGradX / ConvGradW shape
DW = dict(
    N=1, C=64, Hi=12, Wi=12, Ho=12, Wo=12,
    P=3, Q=3, stride=1, pad=1, group=64,
)

# PW ConvGradX / ConvGradW shape
PW = dict(
    N=1, C_in=64, C_out=128, Hi=12, Wi=12, Ho=12, Wo=12,
    P=1, Q=1, stride=1, pad=0, group=1,
)


def _make_conv_grad_x_onnx(
    dY_shape: Sequence[int],
    W_shape: Sequence[int],
    dX_shape: Sequence[int],
    kernel: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],   # [top, left, bottom, right] (ONNX-style)
    group: int,
) -> bytes:
    """Build an ONNX model with a single ConvGradX node (dX = grad w.r.t. input)."""
    dY = helper.make_tensor_value_info("output_grad", TensorProto.FLOAT, list(dY_shape))
    W  = helper.make_tensor_value_info("weight",      TensorProto.FLOAT, list(W_shape))
    dX = helper.make_tensor_value_info("input_grad",  TensorProto.FLOAT, list(dX_shape))
    node = helper.make_node(
        "ConvGradX",
        inputs=["output_grad", "weight"],
        outputs=["input_grad"],
        name="convgradx_node",
        kernel_shape=list(kernel),
        strides=list(strides),
        pads=list(pads),
        dilations=[1, 1],
        group=group,
    )
    graph = helper.make_graph([node], "convgradx_graph", [dY, W], [dX])
    model = helper.make_model(graph,
                              producer_name="convgradx_test",
                              opset_imports=[helper.make_opsetid("", 13)])
    # Fix output shape dims (Deeploy wants them concrete)
    dimlist = list(dX_shape)
    del model.graph.output[0].type.tensor_type.shape.dim[:]
    for d in dimlist:
        model.graph.output[0].type.tensor_type.shape.dim.add().dim_value = d
    return model.SerializeToString()


def _make_conv_grad_w_onnx(
    dY_shape: Sequence[int],
    X_shape: Sequence[int],
    dW_shape: Sequence[int],
    kernel: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    group: int,
) -> bytes:
    dY = helper.make_tensor_value_info("output_grad", TensorProto.FLOAT, list(dY_shape))
    X  = helper.make_tensor_value_info("input_data",  TensorProto.FLOAT, list(X_shape))
    dW = helper.make_tensor_value_info("weight_grad", TensorProto.FLOAT, list(dW_shape))
    node = helper.make_node(
        "ConvGradW",
        inputs=["output_grad", "input_data"],
        outputs=["weight_grad"],
        name="convgradw_node",
        kernel_shape=list(kernel),
        strides=list(strides),
        pads=list(pads),
        dilations=[1, 1],
        group=group,
    )
    graph = helper.make_graph([node], "convgradw_graph", [dY, X], [dW])
    model = helper.make_model(graph,
                              producer_name="convgradw_test",
                              opset_imports=[helper.make_opsetid("", 13)])
    dimlist = list(dW_shape)
    del model.graph.output[0].type.tensor_type.shape.dim[:]
    for d in dimlist:
        model.graph.output[0].type.tensor_type.shape.dim.add().dim_value = d
    return model.SerializeToString()


def _pads_onnx_ordered(pad: int) -> List[int]:
    """Single-value pad → [top, left, bottom, right] (ONNX order)."""
    return [pad, pad, pad, pad]


def build_dw_gradx(out_dir: Path) -> None:
    p = DW
    dY = np.random.randn(p["N"], p["C"], p["Ho"], p["Wo"]).astype(np.float32)
    X  = np.random.randn(p["N"], p["C"], p["Hi"], p["Wi"]).astype(np.float32)
    W  = np.random.randn(p["C"], 1,      p["P"],  p["Q"] ).astype(np.float32)

    # PyTorch reference dX
    dX = torch.nn.grad.conv2d_input(
        input_size=(p["N"], p["C"], p["Hi"], p["Wi"]),
        weight=torch.from_numpy(W),
        grad_output=torch.from_numpy(dY),
        stride=p["stride"], padding=p["pad"], dilation=1, groups=p["group"],
    ).numpy().astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_x_onnx(
            dY.shape, W.shape, dX.shape,
            kernel=[p["P"], p["Q"]], strides=[p["stride"]]*2,
            pads=_pads_onnx_ordered(p["pad"]), group=p["group"],
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "weight": W})
    np.savez(out_dir / "outputs.npz", **{"input_grad": dX})
    print(f"[{out_dir.name}] dY{dY.shape}  W{W.shape}  dX{dX.shape}  "
          f"group={p['group']}  stride={p['stride']}  pad={p['pad']}")


def build_dw_gradw(out_dir: Path) -> None:
    p = DW
    dY = np.random.randn(p["N"], p["C"], p["Ho"], p["Wo"]).astype(np.float32)
    X  = np.random.randn(p["N"], p["C"], p["Hi"], p["Wi"]).astype(np.float32)
    dW = torch.nn.grad.conv2d_weight(
        input=torch.from_numpy(X),
        weight_size=(p["C"], 1, p["P"], p["Q"]),
        grad_output=torch.from_numpy(dY),
        stride=p["stride"], padding=p["pad"], dilation=1, groups=p["group"],
    ).numpy().astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_w_onnx(
            dY.shape, X.shape, dW.shape,
            kernel=[p["P"], p["Q"]], strides=[p["stride"]]*2,
            pads=_pads_onnx_ordered(p["pad"]), group=p["group"],
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "input_data": X})
    np.savez(out_dir / "outputs.npz", **{"weight_grad": dW})
    print(f"[{out_dir.name}] dY{dY.shape}  X{X.shape}  dW{dW.shape}  "
          f"group={p['group']}")


def build_pw_gradx(out_dir: Path) -> None:
    p = PW
    dY = np.random.randn(p["N"], p["C_out"], p["Ho"], p["Wo"]).astype(np.float32)
    X  = np.random.randn(p["N"], p["C_in"],  p["Hi"], p["Wi"]).astype(np.float32)
    W  = np.random.randn(p["C_out"], p["C_in"], p["P"], p["Q"]).astype(np.float32)

    dX = torch.nn.grad.conv2d_input(
        input_size=(p["N"], p["C_in"], p["Hi"], p["Wi"]),
        weight=torch.from_numpy(W),
        grad_output=torch.from_numpy(dY),
        stride=p["stride"], padding=p["pad"], dilation=1, groups=p["group"],
    ).numpy().astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_x_onnx(
            dY.shape, W.shape, dX.shape,
            kernel=[p["P"], p["Q"]], strides=[p["stride"]]*2,
            pads=_pads_onnx_ordered(p["pad"]), group=p["group"],
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "weight": W})
    np.savez(out_dir / "outputs.npz", **{"input_grad": dX})
    print(f"[{out_dir.name}] dY{dY.shape}  W{W.shape}  dX{dX.shape}  PW")


def build_pw_gradw(out_dir: Path) -> None:
    p = PW
    dY = np.random.randn(p["N"], p["C_out"], p["Ho"], p["Wo"]).astype(np.float32)
    X  = np.random.randn(p["N"], p["C_in"],  p["Hi"], p["Wi"]).astype(np.float32)
    dW = torch.nn.grad.conv2d_weight(
        input=torch.from_numpy(X),
        weight_size=(p["C_out"], p["C_in"], p["P"], p["Q"]),
        grad_output=torch.from_numpy(dY),
        stride=p["stride"], padding=p["pad"], dilation=1, groups=p["group"],
    ).numpy().astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_w_onnx(
            dY.shape, X.shape, dW.shape,
            kernel=[p["P"], p["Q"]], strides=[p["stride"]]*2,
            pads=_pads_onnx_ordered(p["pad"]), group=p["group"],
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "input_data": X})
    np.savez(out_dir / "outputs.npz", **{"weight_grad": dW})
    print(f"[{out_dir.name}] dY{dY.shape}  X{X.shape}  dW{dW.shape}  PW")


def build_pw_gradx_sym(out_dir: Path, C: int = 64, HW: int = 12) -> None:
    """Symmetric C_in == C_out PW dX test — isolates the 'fix only worked for
    symmetric' hypothesis in ConvGrad.c:783."""
    dY = np.random.randn(1, C, HW, HW).astype(np.float32)
    W  = np.random.randn(C, C, 1, 1).astype(np.float32)
    dX = torch.nn.grad.conv2d_input(
        input_size=(1, C, HW, HW),
        weight=torch.from_numpy(W),
        grad_output=torch.from_numpy(dY),
        stride=1, padding=0, dilation=1, groups=1,
    ).numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_x_onnx(
            dY.shape, W.shape, dX.shape,
            kernel=[1, 1], strides=[1, 1],
            pads=[0, 0, 0, 0], group=1,
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "weight": W})
    np.savez(out_dir / "outputs.npz", **{"input_grad": dX})
    print(f"[{out_dir.name}] dY{dY.shape}  W{W.shape}  dX{dX.shape}  PW-symmetric")


def build_pw_gradw_sym(out_dir: Path, C: int = 64, HW: int = 12) -> None:
    dY = np.random.randn(1, C, HW, HW).astype(np.float32)
    X  = np.random.randn(1, C, HW, HW).astype(np.float32)
    dW = torch.nn.grad.conv2d_weight(
        input=torch.from_numpy(X),
        weight_size=(C, C, 1, 1),
        grad_output=torch.from_numpy(dY),
        stride=1, padding=0, dilation=1, groups=1,
    ).numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_w_onnx(
            dY.shape, X.shape, dW.shape,
            kernel=[1, 1], strides=[1, 1],
            pads=[0, 0, 0, 0], group=1,
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "input_data": X})
    np.savez(out_dir / "outputs.npz", **{"weight_grad": dW})
    print(f"[{out_dir.name}] dY{dY.shape}  X{X.shape}  dW{dW.shape}  PW-symmetric")


def build_pw_gradx_tiny(out_dir: Path) -> None:
    """Tiny PW dX test (C_in=2, C_out=3, H=W=2) for hand-debugging."""
    C_in, C_out, HW = 2, 3, 2
    dY = np.random.randn(1, C_out, HW, HW).astype(np.float32)
    W  = np.random.randn(C_out, C_in, 1, 1).astype(np.float32)
    dX = torch.nn.grad.conv2d_input(
        input_size=(1, C_in, HW, HW),
        weight=torch.from_numpy(W),
        grad_output=torch.from_numpy(dY),
        stride=1, padding=0, dilation=1, groups=1,
    ).numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_x_onnx(
            dY.shape, W.shape, dX.shape,
            kernel=[1, 1], strides=[1, 1],
            pads=[0, 0, 0, 0], group=1,
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "weight": W})
    np.savez(out_dir / "outputs.npz", **{"input_grad": dX})
    print(f"[{out_dir.name}] dY{dY.shape}  W{W.shape}  dX{dX.shape}  PW-tiny")


def build_pw_gradx_asym_notile(out_dir: Path,
                                C_in: int = 32, C_out: int = 48, HW: int = 8) -> None:
    """Asymmetric PW dX with shape small enough to fit in L1 → no tiling.
    Isolates kernel correctness from tiler bugs."""
    dY = np.random.randn(1, C_out, HW, HW).astype(np.float32)
    W  = np.random.randn(C_out, C_in, 1, 1).astype(np.float32)
    dX = torch.nn.grad.conv2d_input(
        input_size=(1, C_in, HW, HW),
        weight=torch.from_numpy(W),
        grad_output=torch.from_numpy(dY),
        stride=1, padding=0, dilation=1, groups=1,
    ).numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_x_onnx(
            dY.shape, W.shape, dX.shape,
            kernel=[1, 1], strides=[1, 1],
            pads=[0, 0, 0, 0], group=1,
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "weight": W})
    np.savez(out_dir / "outputs.npz", **{"input_grad": dX})
    print(f"[{out_dir.name}] dY{dY.shape}  W{W.shape}  dX{dX.shape}  PW-asym-notile")


def build_pw_gradw_asym_notile(out_dir: Path,
                                C_in: int = 32, C_out: int = 48, HW: int = 8) -> None:
    dY = np.random.randn(1, C_out, HW, HW).astype(np.float32)
    X  = np.random.randn(1, C_in,  HW, HW).astype(np.float32)
    dW = torch.nn.grad.conv2d_weight(
        input=torch.from_numpy(X),
        weight_size=(C_out, C_in, 1, 1),
        grad_output=torch.from_numpy(dY),
        stride=1, padding=0, dilation=1, groups=1,
    ).numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_w_onnx(
            dY.shape, X.shape, dW.shape,
            kernel=[1, 1], strides=[1, 1],
            pads=[0, 0, 0, 0], group=1,
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "input_data": X})
    np.savez(out_dir / "outputs.npz", **{"weight_grad": dW})
    print(f"[{out_dir.name}] dY{dY.shape}  X{X.shape}  dW{dW.shape}  PW-asym-notile")


def build_pw_gradw_tiny(out_dir: Path) -> None:
    C_in, C_out, HW = 2, 3, 2
    dY = np.random.randn(1, C_out, HW, HW).astype(np.float32)
    X  = np.random.randn(1, C_in,  HW, HW).astype(np.float32)
    dW = torch.nn.grad.conv2d_weight(
        input=torch.from_numpy(X),
        weight_size=(C_out, C_in, 1, 1),
        grad_output=torch.from_numpy(dY),
        stride=1, padding=0, dilation=1, groups=1,
    ).numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "network.onnx", "wb") as f:
        f.write(_make_conv_grad_w_onnx(
            dY.shape, X.shape, dW.shape,
            kernel=[1, 1], strides=[1, 1],
            pads=[0, 0, 0, 0], group=1,
        ))
    np.savez(out_dir / "inputs.npz",  **{"output_grad": dY, "input_data": X})
    np.savez(out_dir / "outputs.npz", **{"weight_grad": dW})
    print(f"[{out_dir.name}] dY{dY.shape}  X{X.shape}  dW{dW.shape}  PW-tiny")


def main() -> None:
    np.random.seed(42)
    base = Path(__file__).resolve().parent.parent / "DeeployTest" / "Tests" / "Kernels" / "FP32"
    build_dw_gradx(base / "ConvGradX_DW")
    build_dw_gradw(base / "ConvGradW_DW")
    build_pw_gradx(base / "ConvGradX_PW")
    build_pw_gradw(base / "ConvGradW_PW")
    build_pw_gradx_sym(base / "ConvGradX_PW_sym")
    build_pw_gradw_sym(base / "ConvGradW_PW_sym")
    build_pw_gradx_tiny(base / "ConvGradX_PW_tiny")
    build_pw_gradw_tiny(base / "ConvGradW_PW_tiny")
    build_pw_gradx_asym_notile(base / "ConvGradX_PW_asym_small")
    build_pw_gradw_asym_notile(base / "ConvGradW_PW_asym_small")


if __name__ == "__main__":
    main()
