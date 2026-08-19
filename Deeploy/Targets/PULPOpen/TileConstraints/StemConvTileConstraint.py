# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme

from .ConvTileConstraint import Conv2DTileConstraint


class RQStemConv2DTileConstraint(TileConstraint):
    """Convolution that both reads and writes channels-first.

    Used for a network's first layer, whose input arrives channels-first and
    whose consumer, the depthwise kernel, reads channels-first: leaving the node
    in that layout removes a transpose on either side of it. The reduction runs
    over all input channels, so only the output channels and the spatial axes
    are tiled.
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']
        outputBufferName = parseDict['data_out']

        strides = parseDict["strides"]
        padding = parseDict["pads"]

        for bufferName in [inputBufferName, weightBufferName, mulBufferName, addBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 3)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)
        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputChannelVar == weightOutChannelVar)
        tilerModel.addConstraint(outputChannelVar == addChannelVar)
        tilerModel.addConstraint(outputChannelVar == mulChannelVar)

        inputBuffer = ctxt.lookup(inputBufferName)
        effectiveHeight = inputHeightVar + ((padding[0] + padding[2]) * (inputHeightVar == inputBuffer.shape[2]))
        effectiveWidth = inputWidthVar + ((padding[1] + padding[3]) * (inputWidthVar == inputBuffer.shape[3]))

        tilerModel.addConstraint((outputHeightVar == (effectiveHeight - (weightHeightVar - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint((outputWidthVar == (effectiveWidth - (weightWidthVar - 1) - 1) // strides[1] + 1))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 3)

        strides = parseDict["strides"]
        pads = parseDict["pads"]

        # the kernel accumulates over every input channel, so that axis stays whole
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(weightInChannelVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])

        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'] + pads[0], strategy = None)
        tilerModel.addConstraint(inputWidthVar >= parseDict['dim_kernel_y'] + pads[1], strategy = None)

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        symbolicParseDict = parseDict.copy()
        # the accumulator column above follows the output tile's height
        symbolicParseDict['dim_im_out_x'] = tilerModel.getTensorDimVar(outputBuffer.name, 2)
        symbolicParseDict['dim_im_in_x'] = tilerModel.getTensorDimVar(inputBuffer.name, 2)
        symbolicParseDict['dim_kernel_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 3)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weight', 'mul', 'add', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varWeight = operatorRepresentation['weight']
        varIn = operatorRepresentation['data_in']
        varOut = operatorRepresentation['data_out']

        weightShape = ctxt.lookup(varWeight).shape
        weightC, weightH, weightW = weightShape[1], weightShape[2], weightShape[3]
        inShape = ctxt.lookup(varIn).shape
        outShape = ctxt.lookup(varOut).shape

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": []
        }
        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t)
        }

        inputInCubes = []
        inputAddCubes = []
        inputMulCubes = []
        inputWeightCubes = []

        for cube in outputCubes:
            (BatchOffset, COffset, HOffset, WOffset) = cube.offset
            (BatchSize, CSize, HSize, WSize) = cube.dims

            # computeInputCube speaks channels-last, so hand it the same tile in
            # that order and put the result back
            NHWCOutCube = HyperRectangle((BatchOffset, HOffset, WOffset, COffset), (BatchSize, HSize, WSize, CSize))
            NHWCInCube, padding_tuple = Conv2DTileConstraint.computeInputCube(
                kernelShape = (weightH, weightW),
                pads = pads,
                strides = strides,
                inputCSize = weightC,
                outputCube = NHWCOutCube,
                outputDims = (outShape[0], outShape[2], outShape[3]),
                inputDims = (inShape[0], inShape[2], inShape[3]),
            )
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            NCHWInCube = HyperRectangle((NHWCInCube.offset[0], 0, NHWCInCube.offset[1], NHWCInCube.offset[2]),
                                        (NHWCInCube.dims[0], weightC, NHWCInCube.dims[1], NHWCInCube.dims[2]))

            replacements['dim_im_in_x'].append(NCHWInCube.dims[2])
            replacements['dim_im_in_y'].append(NCHWInCube.dims[3])
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)
            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            RequantCube = HyperRectangle((COffset,), (CSize,))
            inputInCubes.append(NCHWInCube)
            inputWeightCubes.append(HyperRectangle((COffset, 0, 0, 0), (CSize, weightC, weightH, weightW)))
            inputAddCubes.append(RequantCube)
            inputMulCubes.append(RequantCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, b, add, mul in zip(inputInCubes, inputWeightCubes, inputAddCubes, inputMulCubes):
            inputLoadSchedule.append({"data_in": a, "weight": b, "add": add, "mul": mul})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
