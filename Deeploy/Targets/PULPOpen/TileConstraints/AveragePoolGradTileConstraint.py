# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class AveragePoolGradHWTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        # For gradient: data_in is the gradient output (smaller), data_out is gradient input (larger)
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])   # grad_output (smaller)
        outputBuffer = ctxt.lookup(name = parseDict['data_out']) # grad_input (larger)

        strides = parseDict["strides"]
        padding = parseDict["pads"]
        kernelShape = parseDict['kernel_shape']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBuffer.name, outputBuffer.name]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 3)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)  # Batch
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)  # Channel

        # For gradient, reverse the relationship:
        # Forward: output = (input + padding - kernel) / stride + 1
        # Backward: output = (input - 1) * stride + kernel - padding
        # where input is grad_output (small) and output is grad_input (large)

        effectiveHeight = (inputHeightVar - 1) * strides[0] + kernelShape[0]
        effectiveWidth = (inputWidthVar - 1) * strides[1] + kernelShape[1]

        # Remove padding to get the actual output gradient dimensions
        tilerModel.addConstraint(
            outputHeightVar == effectiveHeight - (padding[0] + padding[2]) * (outputHeightVar == outputBuffer.shape[1])
        )
        tilerModel.addConstraint(
            outputWidthVar == effectiveWidth - (padding[1] + padding[3]) * (outputWidthVar == outputBuffer.shape[2])
        )

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        # For gradient: data_out is the gradient input (larger)
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 3)

        strides = parseDict["strides"]

        # Constraint the channel to be full
        tilerModel.addConstraint(outputChannelVar == parseDict['ch_im_out'])

        # For gradient, we need to ensure the output dimensions can be properly divided
        # The output (grad_input) should be divisible by stride to produce valid input (grad_output) tiles
        tilerModel.addConstraint((outputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((outputWidthVar % strides[1]) == 0)

        # Minimum tile size: at least produce one output gradient element
        tilerModel.addConstraint(outputHeightVar >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(outputWidthVar >= parseDict['dim_kernel_y'])

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)
        varIn = operatorRepresentation['data_in']  # grad_output (smaller)

        inputInCubes = []
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

        kernelShape = operatorRepresentation['kernel_shape']
        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        # For gradient: output cubes are grad_input (larger), we need to compute input cubes (grad_output, smaller)
        for cube in outputCubes:
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            # Compute the input cube (grad_output) from output cube (grad_input)
            # This is the reverse of forward pass
            # For gradient, the "input" to the gradient op is smaller
            InHSize = (HSize + pads[0] + pads[2] - (kernelShape[0] - 1) - 1) // strides[0] + 1
            InWSize = (WSize + pads[1] + pads[3] - (kernelShape[1] - 1) - 1) // strides[1] + 1

            InHOffset = HOffset // strides[0]
            InWOffset = WOffset // strides[1]

            InCube = HyperRectangle((BatchOffset, InHOffset, InWOffset, COffset),
                                   (BatchSize, InHSize, InWSize, CSize))

            # Calculate padding for this tile
            padding_top = pads[0] if HOffset == 0 else 0
            padding_bottom = pads[2] if (HOffset + HSize) == ctxt.lookup(operatorRepresentation['data_out']).shape[1] else 0
            padding_left = pads[1] if WOffset == 0 else 0
            padding_right = pads[3] if (WOffset + WSize) == ctxt.lookup(operatorRepresentation['data_out']).shape[2] else 0

            replacements['dim_im_in_x'].append(InHSize)
            replacements['dim_im_in_y'].append(InWSize)
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inputInCubes.append(InCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a in inputInCubes:
            inputLoadSchedule.append({"data_in": a})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
