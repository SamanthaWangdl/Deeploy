# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class PULP2DConvTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(operatorRepresentation['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []

    def computeTransientBuffersSize(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = 2 * 8 * (operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_x'] *
                              operatorRepresentation['dim_kernel_y'])
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = self.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULPStemConvTemplate(PULP2DConvTemplate):

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # PULPStemConv3x3.c sums the input channels through one column of 32-bit
        # accumulators per core, so this follows the output tile's height.
        dim = 4 * 8 * operatorRepresentation['dim_im_out_x']
        name = operatorRepresentation['nodeName'] + "_buffer"
        return [(name, dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # the inherited one asks PULP2DConvTemplate for the size by name, which
        # would hand back the im2col bound instead of the one above
        name, dim = self.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, dim)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = dim
        return ctxt, operatorRepresentation, [name]


class PULP2DDWConvTemplate(PULP2DConvTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(operatorRepresentation['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []

    def computeTransientBuffersSize(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # One column-shaped im2col scratch per core, sized with the kernel's parameters. The call
        # below passes the spatial axes transposed, so dim_in_y is dim_im_in_x and dim_kernel_x is
        # dim_kernel_y.
        pad_top = operatorRepresentation['padding_y_top']
        pad_bot = operatorRepresentation['padding_y_bottom']
        per_core = (operatorRepresentation['dim_kernel_y'] *
                    (operatorRepresentation['dim_im_in_x'] + pad_top + pad_bot) +
                    operatorRepresentation['dim_kernel_y'])
        im2col_dim = 8 * per_core
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]


class PULP1DConvTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(operatorRepresentation['weight'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        operatorRepresentation['pad_x_left'] = operatorRepresentation['pads'][0]
        operatorRepresentation['pad_x_right'] = operatorRepresentation['pads'][1]
        operatorRepresentation['stride_x'] = operatorRepresentation['strides'][0]

        return ctxt, operatorRepresentation, []

    def computeTransientBuffersSize(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = 8 * 2 * operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_y']
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = self.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)
        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULP1DDWConvTemplate(PULP1DConvTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def computeTransientBuffersSize(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # The depthwise pulp-nn kernel reuses one column-shaped im2col scratch
        # per core. Per core it needs `dim_kernel_y * (dim_in_y + pad_top + pad_bot) + dim_kernel_y` bytes (the trailing `+ dim_kernel_y` is the safety zone for the last v4u write).
        pad_top = operatorRepresentation['padding_y_top']
        pad_bot = operatorRepresentation['padding_y_bottom']
        per_core = (operatorRepresentation['dim_kernel_y'] *
                    (operatorRepresentation['dim_im_in_y'] + pad_top + pad_bot) +
                    operatorRepresentation['dim_kernel_y'])
        im2col_dim = 8 * per_core
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]


PULPConv2D_8_Template = PULP2DConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>

<%
operatorString = ''
# A 1x1 kernel makes the im2col copy the identity in HWC, so pulp_nn_pointwise
# indexes the input with the output coordinate and skips it. That only holds
# without stride or padding. Shapes PULPPWConv2D_8_Template handles never reach
# here; this covers the ones its parser turns down.
if (dim_kernel_x == 1 and dim_kernel_y == 1 and stride_x == 1 and stride_y == 1 and
        padding_x_left == 0 and padding_x_right == 0 and padding_y_top == 0 and
        padding_y_bottom == 0):
    kernelName = 'pulp_nn_pointwise' + signatureString
else:
    kernelName = 'pulp_nn_conv' + signatureString
%>

${kernelName}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, ${mul}, ${add}, 1, ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out}, ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_y}, ${stride_x}, 1, 1);
""")

PULPDWConv2D_8_Template = PULP2DDWConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
# PULPDWConv3x3.c specialises 3x3/stride-1/pad-1 and falls back for other shapes.
if signatureString == '_u8_u8_i8':
    kernelName = 'DeeployPULP_DW_Conv2d_3x3_u8_u8_i8'
else:
    kernelName = 'pulp_nn_depthwise' + signatureString
%>
${kernelName}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, NULL, ${mul}, ${add}, 1, ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out}, ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_y}, ${stride_x}, 1, 1);
""")

PULPConv1D_8_Template = PULP1DConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>

pulp_nn_conv${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, ${mul}, ${add}, 1, ${log2D}, 1, ${dim_im_in_y}, ${ch_im_in}, 1, ${dim_im_out_y}, ${ch_im_out}, 1, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, 0, 0, 1, ${stride_y}, 1, 1);
""")

PULPDWConv1D_8_Template = PULP1DDWConvTemplate("""
// PULP NN CONV
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
pulp_nn_depthwise${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, NULL, ${mul}, ${add}, 1, ${log2D}, 1, ${dim_im_in_y}, ${ch_im_in}, 1, ${dim_im_out_y}, ${ch_im_out}, 1, ${dim_kernel_y}, ${padding_y_top}, ${padding_y_bottom}, 0, 0, 1, ${stride_y}, 1, 1);
""")

PULPPWConv2D_8_Template = PULP2DConvTemplate("""
// PULP NN POINTWISE, channels-first output
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
DeeployPULP_PW_Conv2d_1x1_CHWOut${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, ${mul}, ${add}, 1, ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out}, ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_y}, ${stride_x}, 1, 1);
""")

PULPStemConv2D_8_Template = PULPStemConvTemplate("""
// PULP NN CONV, channels-first in and out
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
DeeployPULP_Conv2d_3x3_CHW${signatureString}(${data_in}, ${ctxtBuffer}, NULL, ${data_out}, ${weight}, ${mul}, ${add}, 1, ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out}, ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_y}, ${stride_x}, 1, 1);
""")
