# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import CodeSnippet, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DirectionWaitingStrategy, DmaDirection, Future


class MchanTransferFuture(Future):
    _initTemplate = NodeTemplate("int ${name} = -1;")

    _deinitTemplate = NodeTemplate("")

    _allocTemplate = NodeTemplate("${name} = mchan_transfer_get_id();")

    _waitTemplate = NodeTemplate("""
        if (${name} >= 0) {
            mchan_transfer_wait(${name});
            mchan_transfer_free(${name});
        }
        """)


class GAP9MchanDma(AsyncDma):

    _transferTemplates = {
        1:
            NodeTemplate(
                "{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, .loc = ${loc}, .ext = ${ext} }; mchan_transfer_push_1d(__mchan_tmp); }"
            ),
        2:
            NodeTemplate(
                "{ mchan_transfer_t __mchan_tmp = { .cmd = ${cmd}, .size = ${size}, .loc = ${loc}, .ext = ${ext}, .ext_size_1d = ${size_1d}, .ext_stride_1d = ${stride_2d} }; mchan_transfer_push_2d(__mchan_tmp); }"
            ),
    }
    _waitingStrategy = DirectionWaitingStrategy(MchanTransferFuture, "transfer")

    def __init__(self, transferTemplates: Dict[int, NodeTemplate] = _transferTemplates) -> None:
        super().__init__(transferTemplates)

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        super().checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)

        transferRank = len(shape)
        # MCHAN v7 requires contiguous transfers for innermost dimension in external memory
        assert strideExt[
            -1] == 1, "GAP9 MCHAN supports only contiguous transfers of the innermost dimension for external memory"

        # Local memory (TCDM) must also be contiguous
        if transferRank == 1:
            assert strideLoc[0] == 1, "GAP9 MCHAN supports only contiguous transfers for local memory"
        else:
            assert strideLoc[0] == shape[1] and strideLoc[
                1] == 1, "GAP9 MCHAN supports only contiguous transfers for local memory"

    _MAX_1D_TRANSFER_BYTES = 1 << 17  # 131072 bytes: max representable in 17-bit mchan cmd size field

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        operatorRepresentation = super().transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc,
                                                        direction, future)

        transferRank = len(shape)

        # Build MCHAN command using flags from mchan.h
        # We construct the cmd value in Python and let the C code use the macros
        mchanFlags = 0
        mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0  # direction
        mchanFlags += (1 << 1)  # increment addresses
        mchanFlags += (1 << 2) if transferRank == 2 else 0  # 2d transfer
        mchanFlags += (1 << 3)  # event enable

        mchanTransferSize = math.prod(shape)
        assert mchanTransferSize <= self._MAX_1D_TRANSFER_BYTES, (
            "The transfer size is not representable with 17 bits. "
            f"Received transfer size {mchanTransferSize} that requires "
            f"{math.ceil(math.log2(mchanTransferSize))} bits")

        # cmd = (flags << 17) + size, matching PULPOpen MchanDma pattern
        operatorRepresentation["cmd"] = (mchanFlags << 17) + mchanTransferSize
        operatorRepresentation["size"] = mchanTransferSize

        if transferRank == 2:
            operatorRepresentation["size_1d"] = shape[1]
            operatorRepresentation["stride_2d"] = strideExt[0]

        return operatorRepresentation

    def transfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                 shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                 direction: DmaDirection, future: Future) -> List[CodeSnippet]:
        # For 1D transfers that exceed the 17-bit mchan size field, split into
        # chunked contiguous sub-transfers — mirrors the Siracusa MchanDma
        # fallback so MobileNetV1-style large weight loads still fit GAP9.
        totalSize = math.prod(shape)
        if len(shape) == 1 and totalSize > self._MAX_1D_TRANSFER_BYTES:
            mchanFlags = 0
            mchanFlags += (1 << 0) if direction == "ExternalToLocal" else 0
            mchanFlags += (1 << 1)  # increment addresses
            mchanFlags += (1 << 3)  # event enable
            template = self._transferTemplates[1]
            # Explicitly mangle the buffer names: see Siracusa MchanDma for the
            # same fix — _mangleOpRepr only rewrites plain buffer names, so a
            # string like "((char*)foo + 0)" would ship without the
            # DeeployNetwork_ prefix and fail to compile.
            locName = ctxt._mangle(localBuffer.name)
            extName = ctxt._mangle(externalBuffer.name)
            chunks: List[CodeSnippet] = []
            offset = 0
            while offset < totalSize:
                chunkSize = min(self._MAX_1D_TRANSFER_BYTES, totalSize - offset)
                cmd = (mchanFlags << 17) + chunkSize
                opRepr: OperatorRepresentation = {
                    "loc": f"((char*){locName} + {offset})",
                    "ext": f"((char*){extName} + {offset})",
                    "future": future.name,
                    "cmd": cmd,
                    "size": chunkSize,
                }
                chunks.append(CodeSnippet(template, opRepr))
                offset += chunkSize
            return chunks
        return super().transfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)
