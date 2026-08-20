# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, SequentialPass, \
    contextagnostic
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle


def _singleNodePattern(op: str) -> gs.Graph:
    tensorIn = gs.Variable("input")
    tensorOut = gs.Variable("output")
    node = gs.Node(op = op, name = "node", inputs = [tensorIn], outputs = [tensorOut])
    graph = gs.Graph([node], inputs = [tensorIn], outputs = [tensorOut])
    return graph


def _isDepthwise(node: gs.Node) -> bool:
    if node.op not in ["Conv", "RequantizedConv"]:
        return False

    channels_first = node.attrs.get("channels_first", True)
    spatialDims = len(node.inputs[1].shape) - 2
    shapeIn = node.inputs[0].shape
    chIn = shapeIn[-spatialDims - 1] if channels_first else shapeIn[-1]
    return chIn != 1 and node.attrs.get("group", 1) == chIn


def _createReshape(tensorIn: gs.Tensor,
                   name: str,
                   newShape: Sequence[Union[int, str]],
                   tensorOut: Optional[gs.Tensor] = None) -> Tuple[gs.Node, gs.Tensor]:
    newShapeConst = gs.Constant(name = name + tensorIn.name + "_NewShape", values = np.array(newShape))

    if tensorOut is None:
        tensorOut = gs.Variable(name = name + tensorIn.name + "_Reshaped", dtype = np.float32, shape = newShape)
    else:
        assert newShape == tensorOut.shape

    reshapeNode = gs.Node(name = name + tensorIn.name + "_Reshape",
                          op = "Reshape",
                          inputs = [tensorIn, newShapeConst],
                          outputs = [tensorOut])

    return reshapeNode, tensorOut


def _appendExpandDims(tensor: gs.Tensor, name: str, axis: Union[int, Sequence[int]]) -> Tuple[gs.Node, gs.Tensor]:
    if isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, tuple):
        axes = list(axis)
    elif isinstance(axis, list):
        axes = axis
    else:
        assert False, f"axis should be of type int or tuple. Got {type(axis)}"

    axes = [(len(tensor.shape) + len(axes) + axis if axis < 0 else axis) for axis in axes]
    assert all(axis >= 0 for axis in axes)

    assert isinstance(tensor.shape, Sequence) and len(tensor.shape) > 0 and isinstance(tensor.shape[0], int)
    assert all(axis < len(tensor.shape) + len(axes) for axis in axes), f"axis out of bounds. axis: {axes}"

    newShape = np.zeros(shape = (len(tensor.shape) + len(axes),), dtype = np.int_)
    for axis in axes:
        newShape[axis] = 1
    newShape[newShape == 0] = tensor.shape

    return _createReshape(tensor, name, newShape.tolist())


def _prependSqueezeDims(tensor: gs.Tensor, name: str, axis: Union[int, Sequence[int]]) -> Tuple[gs.Node, gs.Tensor]:
    if isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, tuple):
        axes = list(axis)
    elif isinstance(axis, list):
        axes = axis
    else:
        assert False, f"axis should be of type int or tuple. Got {type(axis)}"

    axes = [(len(tensor.shape) + axis if axis < 0 else axis) for axis in axes]
    assert all(axis >= 0 for axis in axes)

    assert isinstance(tensor.shape, Sequence) and len(tensor.shape) > 0 and isinstance(tensor.shape[0], int)
    assert all(axis < len(tensor.shape) + len(axes) for axis in axes), f"axis out of bounds. axis: {axes}"

    oldShape = np.zeros(shape = (len(tensor.shape) + len(axes),), dtype = np.int_)
    for axis in axes:
        oldShape[axis] = 1
    oldShape[oldShape == 0] = tensor.shape

    inputTensor = gs.Variable(name = name + tensor.name + "_Expanded", dtype = np.float32, shape = oldShape.tolist())

    reshapeNode, _ = _createReshape(inputTensor, name, tensor.shape, tensor)

    return reshapeNode, inputTensor


# Permute (0,1,2,3,...,N-2,N-1) -> (0,1,2,3,...,N-1,N-2)
def _swapLastTwoDimsPermutation(N: int) -> List[int]:
    assert N >= 2, "N needs to be larger than 2"
    return [*range(N - 2), N - 1, N - 2]


# Permute channels first <-> channels last:
#   (*<batch dims>, ch, *<spatial dims>) <-> (*<batch dims>, *<spatial dims>, ch)
def _transformLayoutPermutation(dims: int, spatialDims: int, targetChannelsFirst: bool) -> List[int]:
    batchDims = dims - spatialDims - 1
    if targetChannelsFirst:
        ch = dims - 1
        nonBatchPerm = [ch, *range(batchDims, ch)]
    else:
        ch = batchDims
        nonBatchPerm = [*range(ch + 1, dims), ch]
    return list(range(batchDims)) + nonBatchPerm


# Calculate permutation q = p^(-1) s.t. q(p(i)) = i
def _invertPermutation(permutation: Sequence[int]) -> List[int]:
    return [permutation.index(i) for i in range(len(permutation))]


T = TypeVar('T')


def _permute(_list: Sequence[T], permutation: Sequence[int]) -> List[T]:
    assert len(_list) == len(permutation), "Permuted list and permutation must have equal length!"
    return [_list[i] for i in permutation]


def _permuteHyperRectangle(rect: HyperRectangle, permutation: List[int]) -> HyperRectangle:
    assert len(rect.dims) == len(permutation), "Permutation list and HyperRectangle must have equal dimensionality!"
    return HyperRectangle(tuple(_permute(rect.offset, permutation)), tuple(_permute(rect.dims, permutation)))


def _prependTranspose(tensor: gs.Variable, prevNode: gs.Node, perm: List[int]) -> gs.Node:
    prevNodeTensorIdx = prevNode.outputs.index(tensor)
    preTransposeTensor = gs.Variable(f"{prevNode.name}_{tensor.name}_pre_transposed", tensor.dtype,
                                     _permute(tensor.shape, _invertPermutation(perm)))
    transposeNode = gs.Node(op = "Transpose",
                            name = f"{prevNode.name}_{tensor.name}_pre_transpose",
                            attrs = {"perm": perm},
                            inputs = [preTransposeTensor],
                            outputs = [tensor])
    prevNode.outputs[prevNodeTensorIdx] = preTransposeTensor
    return transposeNode


def _appendTranspose(tensor: gs.Variable, nextNode: gs.Node, perm: List[int]) -> gs.Node:
    nextNodeTensorIdx = nextNode.inputs.index(tensor)
    transposedTensor = gs.Variable(f"{nextNode.name}_{tensor.name}_transposed", tensor.dtype,
                                   _permute(tensor.shape, perm))
    transposeNode = gs.Node(op = "Transpose",
                            name = f"{nextNode.name}_{tensor.name}_transpose",
                            attrs = {"perm": perm},
                            inputs = [tensor],
                            outputs = [transposedTensor])
    nextNode.inputs[nextNodeTensorIdx] = transposedTensor
    return transposeNode


def _transformLayoutConst(const: gs.Constant, spatialDims: int, targetChannelsFirst: bool) -> None:
    assert isinstance(const, gs.Constant)
    if len(const.shape) < 2:
        return
    perm = _transformLayoutPermutation(len(const.shape), spatialDims, targetChannelsFirst)
    const.values = const.values.transpose(perm)


def _transformLayoutDwWeightConst(const: gs.Constant, targetChannelsFirst: bool) -> None:
    assert not targetChannelsFirst, "Target layout should be channels_last!"
    assert isinstance(const, gs.Constant)
    dims = len(const.shape)
    perm = [*range(1, dims), 0]
    const.values = const.values.transpose(perm)


def _transposeMatMulInputs_fun(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    node.attrs['transA'] = node.attrs.get('transA', 0)
    node.attrs['transB'] = node.attrs.get('transB', 0)
    node.attrs['alpha'] = node.attrs.get('alpha', 1.0)
    node.attrs['beta'] = node.attrs.get('beta', 1.0)

    # Prepend transpose on A if it's transposed
    if node.attrs['transA'] == 1:
        tensorA = node.inputs[0]
        perm = _swapLastTwoDimsPermutation(len(tensorA.shape))
        graph.nodes.append(_appendTranspose(tensorA, node, perm))
        node.attrs['transA'] = False

    # Prepend transpose on B if it's not transposed
    if node.attrs['transB'] == 0:
        tensorB = node.inputs[1]
        perm = _swapLastTwoDimsPermutation(len(tensorB.shape))
        graph.nodes.append(_appendTranspose(tensorB, node, perm))
        node.attrs['transB'] = True

    return graph


# SCHEREMO:
# Implements generation of tranpose nodes such that each GEMM/MatMul node implements attributes transA = 0 transB = 1
@contextagnostic
class TransposeMatmulInputsPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("RequantizedGemm")
        name = "_TRANSPOSE_MATMUL_INPUTS_PASS"
        super().__init__(graph, _transposeMatMulInputs_fun, name)


def _NCHWtoNHWC_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):
    node = next(iter((match.nodes_map.values())))

    channels_first = node.attrs.get("channels_first", True)
    if (channels_first != default_channels_first):
        tensorIn = node.inputs[0]
        tensorOut = node.outputs[0]

        if node.op in ["RequantizedConv", "Conv"]:
            spatialDims = len(node.inputs[1].shape) - 2
        elif node.op == "MaxPool":
            spatialDims = len(node.attrs["kernel_shape"])
        elif node.op == "Pad":
            spatialDims = 2  # Hack based on current status
        else:
            raise ValueError(f"Cannot determine spatialDims for node {node.name} with operator {node.op}")

        permuteIn = _transformLayoutPermutation(len(tensorIn.shape), spatialDims, default_channels_first)
        graph.nodes.append(_appendTranspose(tensorIn, node, permuteIn))

        permuteOut = _transformLayoutPermutation(len(tensorOut.shape), spatialDims, channels_first)
        graph.nodes.append(_prependTranspose(tensorOut, node, permuteOut))

        if node.op in ["Conv", "RequantizedConv"]:
            # In the case of Conv: [weights, opt. bias], RequantizedConv: [weights, mul, add, opt. shift]
            for tensor in node.inputs[1:]:
                _transformLayoutConst(tensor, spatialDims, default_channels_first)

        node.attrs["channels_first"] = default_channels_first

    return graph


@contextagnostic
class NCHWtoNHWCMaxPoolPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "MaxPool")
        name = "_NCHW_TO_NHWC_MAXPOOL_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Conv|RequantizedConv")
        name = "_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name,
                         NonBranchingMatcher(regex_op = True))


@contextagnostic
class NCHWtoNHWCPadPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Pad")
        name = "_NCHW_TO_NHWC_PAD_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


def _NCWHtoNHWC_dw_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool) -> gs.Graph:
    node = next(iter((match.nodes_map.values())))

    if not _isDepthwise(node):
        return graph

    channels_first = node.attrs.get("channels_first", True)
    if (channels_first != default_channels_first):
        tensorIn = node.inputs[0]
        tensorOut = node.outputs[0]

        spatialDims = len(node.inputs[1].shape) - 2

        permuteIn = _transformLayoutPermutation(len(tensorIn.shape), spatialDims, default_channels_first)
        permuteOut = _transformLayoutPermutation(len(tensorOut.shape), spatialDims, channels_first)

        graph.nodes.append(_appendTranspose(tensorIn, node, permuteIn))
        graph.nodes.append(_prependTranspose(tensorOut, node, permuteOut))

        _transformLayoutDwWeightConst(node.inputs[1], default_channels_first)  # weights

        if len(node.inputs) > 2:
            # In the case of Conv: [opt. bias], RequantizedConv: [mul, add, opt. shift]
            for tensor in node.inputs[2:]:
                _transformLayoutConst(tensor, spatialDims, default_channels_first)  # bias

        node.attrs["channels_first"] = default_channels_first

    return graph


@contextagnostic
class NCHWtoNHWCDwConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Conv|RequantizedConv")
        name = "_NCHW_TO_NHWC_DW_CONV_PASS"
        super().__init__(graph, partial(_NCWHtoNHWC_dw_fun, default_channels_first = default_channels_first), name,
                         NonBranchingMatcher(regex_op = True))


def _PULP_NCHWtoNHWC_dw_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):
    node = next(iter((match.nodes_map.values())))

    if not _isDepthwise(node):
        return graph

    channels_first = node.attrs.get("channels_first", True)
    if (channels_first != default_channels_first):
        tensorOut = node.outputs[0]

        spatialDims = len(node.inputs[1].shape) - 2

        # LMACAN: PULP DW doesn't transpose the input

        permuteOut = _transformLayoutPermutation(len(tensorOut.shape), spatialDims, channels_first)
        graph.nodes.append(_prependTranspose(tensorOut, node, permuteOut))

        # RequantizedConv: [weights, mul, add, opt. shift]
        for tensor in node.inputs[1:]:
            _transformLayoutConst(tensor, spatialDims, default_channels_first)

        node.attrs["channels_first"] = default_channels_first

    return graph


def _hasUnsignedInput(node: gs.Node) -> bool:
    # The kernel this gate selects takes u8 activations. Types are not resolved
    # yet here, so the producer's `signed` attribute is the only evidence, and an
    # input with no producer -- a graph input, a constant -- has to count as no:
    # committing the layout for a node whose binding then does not exist leaves a
    # graph the parser cannot map at all, since the output transpose is gone.
    # Walk back through the transposes the earlier layout passes inserted: the
    # producer that carries `signed` is the compute node behind them.
    tensor = node.inputs[0]
    for _ in range(4):
        producers = tensor.inputs
        if len(producers) != 1:
            return False
        if producers[0].op != "Transpose":
            break
        tensor = producers[0].inputs[0]
    else:
        return False
    signed = producers[0].attrs.get("signed")
    values = getattr(signed, "values", signed)
    return values is not None and not np.any(values)


def _hasDepthwiseConsumer(node: gs.Node, channels: int) -> bool:
    # stated without reading a layout: this pass may already have relabelled the
    # consumer while leaving its input channels-first
    consumers = node.outputs[0].outputs
    if len(consumers) != 1 or consumers[0].op not in ["Conv", "RequantizedConv"]:
        return False
    return consumers[0].attrs.get("group", 1) == channels


def _hasUnsignedOutput(node: gs.Node) -> bool:
    # The two predicates below decide a layout before the types are resolved, and their
    # kernels exist for an unsigned output only. A node left channels-first with no
    # kernel to read it is a parse failure, so an absent attribute has to mean no.
    signed = node.attrs.get("signed")
    values = getattr(signed, "values", signed)
    return values is not None and not np.any(values)


def isPULPPointwise(node: gs.Node) -> bool:
    """1x1 / stride 1 / no-pad convolutions the PULP kernel can write channels-first.

    The lowering pass, the parser and the template must agree on this exactly, or
    a node ends up producing one layout and being read as the other.
    """
    if node.op != "RequantizedConv" or node.attrs.get("group", 1) != 1:
        return False
    if not _hasUnsignedOutput(node):
        return False
    if len(node.inputs) < 2 or not isinstance(node.inputs[1], gs.Constant):
        return False
    if list(node.attrs.get("kernel_shape", [])) != [1, 1]:
        return False
    if list(node.attrs.get("strides", [1, 1])) != [1, 1]:
        return False
    if any(pad != 0 for pad in node.attrs.get("pads", [0, 0, 0, 0])):
        return False
    weightShape = node.inputs[1].shape
    if len(weightShape) != 4:
        return False
    # this runs both before and after the layout pass, which rewrites the weight
    # from [Cout, Cin, 1, 1] to [Cout, 1, 1, Cin]
    chIn = weightShape[1] if node.attrs.get("channels_first", True) else weightShape[3]
    if not _hasUnsignedInput(node):
        return False
    # the kernel blocks two output channels and four input bytes at a time
    # No condition on the consumer, unlike isPULPStemConv: only this node's output
    # side moves, so a consumer that wants channels-last just gets one transpose
    # back. That is the cheaper half of the trade -- restricting this to depthwise
    # consumers sends MobileNetV1's last pointwise to pulp_nn_pointwise, whose row
    # split degenerates on its 3x3 output, and costs 37k cycles to save 2k.
    return weightShape[0] % 2 == 0 and chIn % 4 == 0


def _PULP_NCHWtoNHWC_pw_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):
    node = next(iter((match.nodes_map.values())))

    if not isPULPPointwise(node):
        return graph

    channels_first = node.attrs.get("channels_first", True)
    if (channels_first != default_channels_first):
        tensorIn = node.inputs[0]
        spatialDims = len(node.inputs[1].shape) - 2

        permuteIn = _transformLayoutPermutation(len(tensorIn.shape), spatialDims, default_channels_first)
        graph.nodes.append(_appendTranspose(tensorIn, node, permuteIn))

        # the PULP pointwise kernel writes channels-first, so unlike the dense
        # case the output needs no transpose back

        # RequantizedConv: [weights, mul, add, opt. shift]
        for tensor in node.inputs[1:]:
            _transformLayoutConst(tensor, spatialDims, default_channels_first)

        node.attrs["channels_first"] = default_channels_first

    return graph


@contextagnostic
class PULPNCHWtoNHWCPwConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "RequantizedConv")
        name = "_PULP_NCHW_TO_NHWC_PW_CONV_PASS"
        super().__init__(graph, partial(_PULP_NCHWtoNHWC_pw_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class PULPNCHWtoNHWCDwConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "RequantizedConv")
        name = "_PULP_NCHW_TO_NHWC_DW_CONV_PASS"
        super().__init__(graph, partial(_PULP_NCHWtoNHWC_dw_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCPass(SequentialPass):

    def __init__(self, default_channels_first: bool = True):
        passes = [
            NCHWtoNHWCPadPass(default_channels_first),
            NCHWtoNHWCMaxPoolPass(default_channels_first),
            NCHWtoNHWCDwConvPass(default_channels_first),
            NCHWtoNHWCConvPass(default_channels_first),
        ]
        super().__init__(*passes)


def isPULPStemConv(node: gs.Node) -> bool:
    """First-layer convolution that can stay channels-first on both sides.

    Both ends have to want that layout: the input must be a graph input, which
    arrives channels-first, and the consumer must be a depthwise convolution,
    which reads channels-first. Feeding a dense convolution instead would only
    move the transpose rather than remove it. The shape is the one
    PULPStemConv3x3.c has a fast path for; anything else would land in its
    reference loop, which is far slower than the im2col kernel it replaced.
    """
    if node.op != "RequantizedConv" or node.attrs.get("group", 1) != 1:
        return False
    if not _hasUnsignedOutput(node):
        return False
    if len(node.inputs) < 2 or not isinstance(node.inputs[1], gs.Constant):
        return False
    if len(node.inputs[0].inputs) != 0:
        return False
    # spelled out rather than tested for a property: an absent attribute means no
    # padding and unit strides, which the fast path in PULPStemConv3x3.c is not
    # written for -- its rows advance by two and carry only the window's bottom
    # row over, which is the stride-2 overlap
    if list(node.attrs.get("kernel_shape", [])) != [3, 3]:
        return False
    if list(node.attrs.get("pads", [])) != [1, 1, 1, 1]:
        return False
    if list(node.attrs.get("strides", [])) != [2, 2]:
        return False
    weightShape = node.inputs[1].shape
    if len(weightShape) != 4:
        return False
    chIn = weightShape[1] if node.attrs.get("channels_first", True) else weightShape[3]
    if chIn != 3:
        return False
    return _hasDepthwiseConsumer(node, weightShape[0])


def _PULP_NCHWtoNHWC_conv_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):
    node = next(iter((match.nodes_map.values())))

    if isPULPStemConv(node):
        return graph

    return _NCHWtoNHWC_fun(graph, match, name, default_channels_first)


@contextagnostic
class PULPNCHWtoNHWCConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Conv|RequantizedConv")
        name = "_PULP_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_PULP_NCHWtoNHWC_conv_fun, default_channels_first = default_channels_first),
                         name, NonBranchingMatcher(regex_op = True))


@contextagnostic
class PULPNCHWtoNHWCPass(SequentialPass):

    def __init__(self, default_channels_first: bool = True):
        passes = [
            NCHWtoNHWCPadPass(default_channels_first),
            NCHWtoNHWCMaxPoolPass(default_channels_first),
            PULPNCHWtoNHWCDwConvPass(default_channels_first),
            PULPNCHWtoNHWCPwConvPass(default_channels_first),
            PULPNCHWtoNHWCConvPass(default_channels_first),
        ]
        super().__init__(*passes)


def _requantized_gemm_to_pw_fun(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    matrixA: gs.Variable = node.inputs[0]
    matrixB: gs.Constant = node.inputs[1]
    matrixY: gs.Variable = node.outputs[0]

    # Check matrixB is a constant, otherwise don't transform
    if not isinstance(matrixB, gs.Constant):
        return graph

    assert len(matrixA.shape) in (2, 3), f"Unsupported GEMM's input matrix A with shape {matrixA.shape}"
    assert len(matrixY.shape) in (2, 3), f"Unsupported GEMM's output matrix with shape {matrixY.shape}"

    # The pointwise conv this node is about to be lowered into only supports
    # per-tensor or per-output-channel requantization via its RQS unit. If
    # mul/add don't fit that (e.g. per-row requantization, where M becomes a
    # spatial dimension of the convolution rather than a channel), bail out
    # here and leave the RequantizedGemm for another lowering pass to handle.
    add, mul = node.inputs[2], node.inputs[3]
    out_channels = matrixY.shape[-1]
    if int(np.prod(mul.shape)) not in (1, out_channels) or int(np.prod(add.shape)) not in (1, out_channels):
        return graph

    # Pointwise with HWC layout (channels_first == False)

    # Defaults
    node.attrs['transA'] = node.attrs.get('transA', 0)
    node.attrs['transB'] = node.attrs.get('transB', 0)
    node.attrs['alpha'] = node.attrs.get('alpha', 1.0)
    node.attrs['beta'] = node.attrs.get('beta', 1.0)

    # If transA is set then the matrix is of shape [B x K x M] and it needs to
    # be transposed, otherwise its shape is  [B x M x K]
    if node.attrs['transA'] == 1:
        perm = _swapLastTwoDimsPermutation(len(matrixA.shape))
        graph.nodes.append(_appendTranspose(matrixA, node, perm))
        matrixA = node.inputs[0]

    # If transB is set then the matrix is of shape [N x K] and it doesn't need
    # to be transposed, otherwise its shape is [K x N] and it has to be transposed
    if node.attrs['transB'] == 0:
        perm = _swapLastTwoDimsPermutation(len(matrixB.shape))
        matrixB.values = matrixB.values.transpose(perm)

    # Align dimensions for convolution
    expandAxis = []
    # Align the batch dimension
    if len(matrixA.shape) == 2:
        expandAxis.append(0)
    # Expand the height dimension
    expandAxis.append(1)
    # pwIn, shape [B x 1 x M x K]
    matrixAExpandDimsNode, pwIn = _appendExpandDims(matrixA, name, axis = expandAxis)
    graph.nodes.append(matrixAExpandDimsNode)

    # pwWeight, shape [N x 1 x 1 x K]
    matrixBExpandDimsNode, pwWeight = _appendExpandDims(matrixB, name, axis = (1, 2))
    graph.nodes.append(matrixBExpandDimsNode)

    if len(matrixY.shape) == 2:
        # matrixY, shape [M x N]
        squeezeDims = (0, 1)
    else:
        # matrixY, shape [B x M x N]
        squeezeDims = (1,)
    # pwOut, shape [B x 1 x M x N]
    matrixYSqueezeDimsNode, pwOut = _prependSqueezeDims(matrixY, name, squeezeDims)
    graph.nodes.append(matrixYSqueezeDimsNode)

    n_levels = node.attrs['n_levels_out'] if 'n_levels_out' in node.attrs else node.attrs['n_levels']
    pwAttrs = {
        'channels_first': False,
        'dilations': [1, 1],
        'group': 1,
        'kernel_shape': [1, 1],
        'pads': [0, 0, 0, 0],
        'strides': [1, 1],
        'div': node.attrs['div'],
        'n_levels_out': n_levels,
        'shift': node.attrs['shift'],
        'signed': node.attrs['signed'],
    }

    _inputs = [pwIn, pwWeight, mul, add]

    pw = gs.Node(op = 'RequantizedConv',
                 name = name + "_RequantizedPwConv",
                 inputs = _inputs,
                 outputs = [pwOut],
                 attrs = pwAttrs)
    graph.nodes.append(pw)

    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.remove(node)

    return graph


@contextagnostic
class RequantizedGemmToPwPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("RequantizedGemm")
        super().__init__(graph, _requantized_gemm_to_pw_fun, "_REQUANTIZED_GEMM_TO_PW_PASS")


def _remove_global_output_reshape_fun(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    isGlobalOutput = len(node.outputs[0].outputs) == 0
    if isGlobalOutput:
        graph.deleteNode(node)

    return graph


@contextagnostic
class RemoveGlobalOutputReshapePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("Reshape")
        super().__init__(graph, _remove_global_output_reshape_fun, "_REMOVE_GLOBAL_OUTPUT_RESHAPE_PASS")


def _remove_empty_conv_bias_fun(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    # Check if the node has an all-zero bias and remove it
    if len(node.inputs) == 3:
        bias = node.inputs[2]
        if isinstance(bias, gs.Constant) and np.all(bias.values == 0):
            del node.inputs[2]

    return graph


@contextagnostic
class RemoveEmptyConvBiasPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("Conv")
        name = "_REMOVE_EMPTY_CONV_BIAS_PASS"
        super().__init__(graph, _remove_empty_conv_bias_fun, name)


def _remove_only_singleton_reduce_mean(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    # Keep node if only one in the graph
    if len(graph.nodes) == 1:
        return graph

    # Delete node if only reduction over singleton dimensions
    if 'axis' in node.attrs:
        axis = node.attrs['axis']
    else:
        axis = node.inputs[1].values

    # Check if shape information is available
    if node.inputs[0].shape is not None and all(node.inputs[0].shape[ax] == 1 for ax in axis):
        graph.deleteNode(node)

    return graph


@contextagnostic
class RemoveOnlySingletonReduceMeanPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("ReduceMean")
        name = "_REMOVE_ONLY_SINGLETON_REDUCE_MEAN_PASS"
        super().__init__(graph, _remove_only_singleton_reduce_mean, name)
