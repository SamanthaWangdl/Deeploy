# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class RQPWConv2DTileConstraint(TileConstraint):
    """1x1 / stride 1 / no-pad convolution with a channels-first output.

    Mirrors RQDWConv2DTileConstraint, which reads channels-first and writes
    channels-last: together the two alternate without a layout change in
    between. The input keeps the channels-last order the kernel multiplies in,
    so only the output tensor's axes move.
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']
        outputBufferName = parseDict['data_out']

        for bufferName in [inputBufferName, weightBufferName, mulBufferName, addBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)

        # channels-first output
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

        # a 1x1 kernel without stride or padding maps the spatial axes one to one
        tilerModel.addConstraint(outputHeightVar == inputHeightVar)
        tilerModel.addConstraint(outputWidthVar == inputWidthVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 3)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)

        # the kernel accumulates the whole reduction, so it cannot be split
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(weightInChannelVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(weightHeightVar == 1)
        tilerModel.addConstraint(weightWidthVar == 1)

        # two output channels per core keep the 4x2 block whole
        if parseDict["ch_im_out"] >= 2 * 8:
            tilerModel.addMinTileSizeConstraint(parseDict, 'ch_im_out', outputChannelVar, 2 * 8)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['dim_im_in_x'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)

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

        weightC = ctxt.lookup(operatorRepresentation['weight']).shape[3]

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_out": [],
        }
        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
        }

        inputInCubes = []
        inputAddCubes = []
        inputMulCubes = []
        inputWeightCubes = []

        for cube in outputCubes:
            (BatchOffset, COffset, HOffset, WOffset) = cube.offset
            (BatchSize, CSize, HSize, WSize) = cube.dims

            InCube = HyperRectangle((BatchOffset, HOffset, WOffset, 0), (BatchSize, HSize, WSize, weightC))

            replacements['dim_im_in_x'].append(HSize)
            replacements['dim_im_in_y'].append(WSize)
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)

            inputInCubes.append(InCube)
            RequantCube = HyperRectangle((COffset,), (CSize,))
            inputWeightCubes.append(HyperRectangle((COffset, 0, 0, 0), (CSize, 1, 1, weightC)))
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
