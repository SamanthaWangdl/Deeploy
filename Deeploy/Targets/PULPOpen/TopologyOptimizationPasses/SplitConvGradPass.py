# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, contextagnostic


def _classifyOutputs(node: gs.Node, x: gs.Tensor, w: gs.Tensor):
    """Sort a ConvGrad's outputs into (dX, dW, dB) by SHAPE, not by position.

    ORT emits dX only when something downstream needs it. A trainable layer whose input
    needs no gradient -- the first trainable layer of a network -- emits dW alone, and a
    positional read then takes that dW to be dX. Nothing downstream can recover: the
    parser binds it as the input gradient, the tiler slices it along its output channels,
    and ConvGradXTileConstraintBase reads those cube offsets as batch offsets, deriving a
    dY offset of (k, 0, 0, 0) on a tensor whose batch extent is 1.

    dX carries X's shape and dW carries W's shape, which tells them apart whatever order
    they arrive in. Falls back to the positional reading when shapes are unavailable, so
    graphs that never had shape annotations behave exactly as before.
    """
    outs = [o for o in node.outputs if o is not None]
    xShape = list(x.shape) if x.shape is not None else None
    wShape = list(w.shape) if w.shape is not None else None

    if xShape is None or wShape is None or xShape == wShape:
        # No way to tell them apart; keep the historical positional assignment.
        return (outs[0] if len(outs) > 0 else None, outs[1] if len(outs) > 1 else None,
                outs[2] if len(outs) > 2 else None)

    dx = dw = db = None
    for o in outs:
        shape = list(o.shape) if o.shape is not None else None
        if shape == wShape and dw is None:
            dw = o
        elif shape == xShape and dx is None:
            dx = o
        elif shape is not None and len(shape) == 1 and db is None:
            db = o
        elif dx is None and shape is None:
            # An unannotated output can only be placed positionally; dX comes first.
            dx = o
    return dx, dw, db


def _split_single_conv_grad(graph: gs.Graph, node: gs.Node, counter: int):
    """Split one ConvGrad node → ConvGradX + ConvGradW [+ ConvGradB].

    Original ConvGrad:
        inputs : [dY, X, W]           (no bias)
        outputs: [dX, dW]

    or:
        inputs : [dY, X, W, B]        (with bias)
        outputs: [dX, dW, dB]

    After split:
        ConvGradX:  inputs=[dY, W]    → outputs=[dX]
        ConvGradW:  inputs=[dY, X]    → outputs=[dW]
        ConvGradB:  inputs=[dY]       → outputs=[dB]   (only when bias present)
    """
    if len(node.inputs) < 3 or len(node.outputs) < 1:
        return

    dy = node.inputs[0]  # dY: upstream gradient  [N, C_out, H_out, W_out]
    x = node.inputs[1]  # X:  forward input       [N, C_in,  H_in,  W_in]
    w = node.inputs[2]  # W:  weight              [C_out, C_in/group, kH, kW]

    dx, dw, db = _classifyOutputs(node, x, w)

    # Copy attrs; add kernel_shape from the weight tensor to avoid
    # Conv2DParser.parseNode computing wrong kernel_shape from inputs[1].
    attrs_x = dict(node.attrs)
    attrs_w = dict(node.attrs)

    if 'kernel_shape' not in attrs_x and w.shape is not None and len(w.shape) >= 4:
        attrs_x['kernel_shape'] = list(w.shape[2:4])

    base_name = node.name if node.name else f'ConvGrad_{counter}'

    # ConvGradX: compute dX from dY and W. Skipped when the node has no dX at all --
    # a trainable layer whose input needs no gradient emits dW only.
    if dx is not None:
        conv_grad_x = gs.Node(
            op = 'ConvGradX',
            name = f'{base_name}_ConvGradX',
            inputs = [dy, w],
            outputs = [dx],
            attrs = attrs_x,
        )
        graph.nodes.append(conv_grad_x)

    if dw is not None:

        # Propagate shape and dtype from W → dW (same shape; ONNX shape inference misses ConvGrad)
        if dw.shape is None and w.shape is not None:
            dw.shape = list(w.shape)
        if dw.dtype is None and w.dtype is not None:
            dw.dtype = w.dtype

        if 'kernel_shape' not in attrs_w and w.shape is not None and len(w.shape) >= 4:
            attrs_w['kernel_shape'] = list(w.shape[2:4])
        elif 'kernel_shape' not in attrs_w and dw.shape is not None and len(dw.shape) >= 4:
            attrs_w['kernel_shape'] = list(dw.shape[2:4])

        # ConvGradW: compute dW from dY and X
        conv_grad_w = gs.Node(
            op = 'ConvGradW',
            name = f'{base_name}_ConvGradW',
            inputs = [dy, x],
            outputs = [dw],
            attrs = attrs_w,
        )
        graph.nodes.append(conv_grad_w)

        if db is not None:

            # Propagate bias shape and dtype: dB shape == B shape (or [C_out] from W)
            if db.shape is None:
                if len(node.inputs) >= 4 and node.inputs[3].shape is not None:
                    db.shape = list(node.inputs[3].shape)
                elif w.shape is not None:
                    db.shape = [w.shape[0]]
            if db.dtype is None:
                if len(node.inputs) >= 4 and node.inputs[3].dtype is not None:
                    db.dtype = node.inputs[3].dtype
                elif w.dtype is not None:
                    db.dtype = w.dtype

            # ConvGradB: compute dB = sum(dY, axes=[N, H, W])
            conv_grad_b = gs.Node(
                op = 'ConvGradB',
                name = f'{base_name}_ConvGradB',
                inputs = [dy],
                outputs = [db],
                attrs = {},
            )
            graph.nodes.append(conv_grad_b)

    # Remove the original ConvGrad node
    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.remove(node)


@contextagnostic
class SplitConvGradPass(Pass):
    """Replace each ConvGrad node with ConvGradX + ConvGradW[B] nodes.

    Handles 1/2/3 outputs:
      1 output (dX only):      ConvGradX
      2 outputs (dX + dW):     ConvGradX + ConvGradW
      3 outputs (dX + dW + dB): ConvGradX + ConvGradW + ConvGradB

    No-op for inference graphs (which have no ConvGrad nodes).
    """

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        # Collect all ConvGrad nodes first (avoid modifying list while iterating)
        nodes_to_split = [n for n in graph.nodes if n.op == 'ConvGrad']

        for counter, node in enumerate(nodes_to_split):
            _split_single_conv_grad(graph, node, counter)

        if nodes_to_split:
            graph.cleanup()

        return graph
