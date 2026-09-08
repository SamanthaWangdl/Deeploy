# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
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


class ConvGradXTileConstraintBase(TileConstraint):
    """
    Base for ConvGradX2D tiling:

      - absoluteOutputCubes are tiles of grad_in (dX)  (operatorRepresentation[gradInKey])
      - for each dX tile, derive required grad_out (dY) halo tile
      - weight is full (not tiled)
      - emits unified template params:
          ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
          ${offset_grad_in_h}, ${offset_grad_in_w}, ${offset_grad_out_h}, ${offset_grad_out_w}
    """

    # ---- parser/opRep keys (override in subclasses if needed) ----
    # In Deeploy ConvGradX parsers these are commonly "data_in" (dY) and "data_out" (dX).
    gradOutKey = "grad_out"  # dY
    gradInKey = "grad_in"  # dX
    weightKey = "weight"  # W

    # ---------------------------
    # 1) Geometrical constraints
    # ---------------------------
    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]
        wName = parseDict[cls.weightKey]

        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dxName)
        tilerModel.addTensorDimToModel(ctxt, wName)

        group = parseDict.get("group", 1)

        # N match
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 0) == tilerModel.getTensorDimVar(dxName, 0))

        # Channel relations:
        # dY: [N, C_out, H_out, W_out]
        # dX: [N, C_in,  H_in,  W_in]
        # W : [C_out, C_in/group, P, Q]
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == tilerModel.getTensorDimVar(wName, 0))
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 1) == tilerModel.getTensorDimVar(wName, 1) * group)

        return tilerModel

    # -----------------------
    # 2) Policy constraints
    # -----------------------
    # Minimum arithmetic intensity target for ConvGradX tiles.
    # Below this AI, the tile is severely memory-bound and DMA dominates.
    _AI_MIN_TARGET = 4.0

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Cost-model-aware policy for ConvGradX tiling:
          - keep full Cout on dY (reduction axis)
          - weight kernel dims (kH, kW) stay full
          - Cin allowed to tile (lockstep with dX)
          - **NEW**: minimum Cin tile size derived from roofline AI target,
            ensuring each tile has enough compute to amortize DMA cost.
        """
        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]
        wName = parseDict[cls.weightKey]

        dyBuf = ctxt.lookup(dyName)
        dxBuf = ctxt.lookup(dxName)
        wBuf = ctxt.lookup(wName)

        # Cout full on dY (reduction axis for ConvGradX)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == dyBuf.shape[1])

        # Weight: C_out full, kH/kW full; Cin free to tile
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 0) == wBuf.shape[0])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 2) == wBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 3) == wBuf.shape[3])

        # ── Cost-model: minimum Cin tile for AI target ──
        # For ConvGradX, per-tile:
        #   FLOPs = 2 × Cout × cin_t × kH × kW × Ho_t × Wo_t
        #   bytes ≈ (Cout × Ho_t × Wo_t + cin_t × Hi_t × Wi_t + Cout × cin_t × kH × kW) × 4
        # At minimum (spatial full, only Cin tiled):
        #   FLOPs = 2 × Cout × cin_t × K × Ho × Wo
        #   bytes ≈ (dY_full + W_tile + dX_tile) = dY + Cout×cin_t×K×4 + cin_t×Hi×Wi×4
        #   AI = FLOPs / bytes
        #   Solve for cin_t such that AI >= AI_MIN_TARGET
        try:
            Cout = wBuf.shape[0]
            Cin = wBuf.shape[1]  # Cin per group
            kH, kW = wBuf.shape[2], wBuf.shape[3]
            K = kH * kW
            N, _, Ho, Wo = dyBuf.shape
            _, _, Hi, Wi = dxBuf.shape
            bpe = 4

            # dY is full (Cout pinned): dY_bytes = N × Cout × Ho × Wo × 4
            dy_bytes = N * Cout * Ho * Wo * bpe
            # Per cin_t: W_tile_bytes = Cout × cin_t × K × 4
            #            dX_tile_bytes = N × cin_t × Hi × Wi × 4
            # tile_flops = 2 × N × Cout × cin_t × K × Ho × Wo
            # AI = tile_flops / (dy_bytes + W_tile + dX_tile)
            #    = 2 × N × Cout × cin_t × K × Ho × Wo / (dy_bytes + cin_t × (Cout×K + N×Hi×Wi) × 4)
            #
            # For AI >= AI_MIN: solve for cin_t
            #   2×N×Cout×cin_t×K×Ho×Wo >= AI_MIN × (dy_bytes + cin_t × per_cin_bytes)
            #   cin_t × (2×N×Cout×K×Ho×Wo - AI_MIN × per_cin_bytes) >= AI_MIN × dy_bytes
            per_cin_bytes = (Cout * K + N * Hi * Wi) * bpe
            flops_per_cin = 2 * N * Cout * K * Ho * Wo
            numerator = cls._AI_MIN_TARGET * dy_bytes
            denominator = flops_per_cin - cls._AI_MIN_TARGET * per_cin_bytes

            l1 = _HW_PARAMS['l1_size']

            # ── W tile size cap: W_tile should not exceed L1/2 ──
            # When W is large (e.g. 147KB > L1), the solver may allocate most
            # of L1 to W_tile, leaving tiny spatial tiles (5×1) → many tiles.
            # Cap W_tile to L1/2 so dY+dX get adequate space for larger spatial tiles.
            w_per_cin = Cout * K * bpe  # bytes per Cin channel in W
            max_cin_for_w = max(1, (l1 // 2) // w_per_cin)
            max_cin_for_w = min(max_cin_for_w, Cin)

            if denominator > 0:
                cin_min = max(1, int(numerator / denominator) + 1)
                cin_min = min(cin_min, Cin)  # don't exceed full
                # Feasibility must win over the AI perf target: never *force* a
                # Cin tile larger than what keeps the weight tile within L1/2.
                # Otherwise large-Cout pointwise layers (e.g. MobileNetV1
                # block_11 PW, Cout=256) get cin_min≈101 → a 103KB weight tile →
                # L1 floor ≈116KB, which exceeds the GAP9 usable L1 heap (≈109KB)
                # and makes the whole network untileable. Cin is NOT a reduction
                # axis for dX (each Cin channel is independent: dX[ci]=Σ_co
                # W[co,ci]·dY[co]), so clamping the Cin tile is always correct —
                # it only trades a little arithmetic intensity for feasibility.
                cin_min = min(cin_min, max_cin_for_w)

                # Check feasibility: W_tile + dY + minimal dX must fit L1
                w_tile_min = Cout * cin_min * K * bpe
                dx_tile_min = N * cin_min * 1 * 1 * bpe  # minimal 1×1 spatial
                if dy_bytes + w_tile_min + dx_tile_min <= l1 and cin_min > 1:
                    tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 1) >= cin_min)
            # else: flops too low relative to bytes, can't hit AI target → no constraint

            # Always bound the weight tile to L1/2 (feasibility), independent of
            # whether the AI minimum applied above.
            if max_cin_for_w < Cin:
                tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 1) <= max_cin_for_w)

            # Note: spatial minimum constraints were considered but cause regression
            # on layers where the solver needs spatial freedom (layer1 with stride=2).
            # The Cin constraint above is sufficient for large-W layers (layer3).
        except Exception:
            pass  # best-effort; don't block tiling on cost model failure

        # RW: For strided convs, force the dX spatial tile size to be a multiple
        # of the stride. A 1x1 stride-s conv samples only every s-th input
        # position, so an input column/row whose index is not a multiple of s
        # receives NO output-gradient contribution. If the tiler isolates such a
        # column into its own tile (e.g. a width-1 tile under tight L1), the
        # inverse-conv dY halo for that tile is empty -> computeDyCubeFromDxTile
        # raises (generation) or the kernel writes past its tile (runtime L1 OOB
        # at update 1). Constraining the dX spatial tile to a multiple of the
        # stride guarantees every tile spans a full stride period (both sampled
        # and unsampled positions), so the halo is never empty. Only applied for
        # stride > 1, so stride-1 layers keep full spatial freedom.
        try:
            dxName = parseDict[cls.gradInKey]
            sh, sw = tuple(parseDict.get("strides", [1, 1]))
            if sh > 1:
                tilerModel.addTileSizeDivisibleConstraint(parseDict, "dim_im_in_x",
                                                          tilerModel.getTensorDimVar(dxName, 2), sh)
            if sw > 1:
                tilerModel.addTileSizeDivisibleConstraint(parseDict, "dim_im_in_y",
                                                          tilerModel.getTensorDimVar(dxName, 3), sw)
        except Exception:
            pass  # best-effort; don't block tiling if keys/strides unavailable

        return tilerModel

    # -----------------------------------
    # 3) Symbolic node representation
    # -----------------------------------
    @classmethod
    def constructSymbolicNodeRep(cls, tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        """
        Bind template fields:
          dim_im_out_* / ch_im_out : grad_out (dY)
          dim_im_in_*  / ch_im_in  : grad_in  (dX)
        """
        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]
        wName = parseDict[cls.weightKey]

        symbolic = parseDict.copy()

        # dY (grad_out)
        symbolic["dim_im_out_x"] = tilerModel.getTensorDimVar(dyName, 2)  # H_out tile
        symbolic["dim_im_out_y"] = tilerModel.getTensorDimVar(dyName, 3)  # W_out tile
        symbolic["ch_im_out"] = tilerModel.getTensorDimVar(dyName, 1)  # Cout

        # dX (grad_in)
        symbolic["dim_im_in_x"] = tilerModel.getTensorDimVar(dxName, 2)  # H_in tile
        symbolic["dim_im_in_y"] = tilerModel.getTensorDimVar(dxName, 3)  # W_in tile
        symbolic["ch_im_in"] = tilerModel.getTensorDimVar(dxName, 1)  # Cin

        # kernel (H,W)
        symbolic["dim_kernel_x"] = tilerModel.getTensorDimVar(wName, 2)  # P
        symbolic["dim_kernel_y"] = tilerModel.getTensorDimVar(wName, 3)  # Q

        # offsets filled in serialize
        symbolic["offset_grad_in_h"] = 0
        symbolic["offset_grad_in_w"] = 0
        symbolic["offset_grad_out_h"] = 0
        symbolic["offset_grad_out_w"] = 0

        return symbolic

    # -------------------------------
    # helpers
    # -------------------------------
    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return -((-a) // b)

    @staticmethod
    def _floor_div(a: int, b: int) -> int:
        return a // b

    @classmethod
    def get_kernel_hw(cls, ctxt: NetworkContext, wName: str, wShape: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return wShape[2], wShape[3]

    @classmethod
    def get_dy_channels(
        cls,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
        dyName: str,
        dxName: str,
        wName: str,
        dyFull: Tuple[int, int, int, int],
        dxFull: Tuple[int, int, int, int],
        wShape: Tuple[int, int, int, int],
    ) -> int:
        # default ConvGradX: dy channels == weight[0] (Cout)
        return wShape[0]

    @classmethod
    def get_ch_im_out(cls, ctxt: NetworkContext, dyFull, dxFull, wShape) -> int:
        # template's ch_im_out should match dY channels (Cout)
        return dyFull[1]

    @classmethod
    def get_ch_im_in(cls, ctxt: NetworkContext, dyFull, dxFull, wShape) -> int:
        # template's ch_im_in should match dX channels (Cin)
        return dxFull[1]

    @classmethod
    def map_onnx_pads_to_template(cls, tpt: int, tpb: int, tpl: int, tpr: int) -> Tuple[int, int, int, int]:
        """
        ONNX pads are (top, bottom, left, right) where top/bottom are H, left/right are W.

        Template unified order:
          (${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right})

        Both tiled C kernels expect:
          - arg1 (${padding_y_top})    -> pad_top   (H_begin)
          - arg2 (${padding_y_bottom}) -> pad_bottom (H_end)
          - arg3 (${padding_x_left})   -> pad_left  (W_begin)
          - arg4 (${padding_x_right})  -> pad_right (W_end)

        So the mapping is the identity: padding_y_top=top, padding_x_left=left.
        """
        return (tpt, tpb, tpl, tpr)

    @classmethod
    def computeDyCubeFromDxTile(
            cls,
            dxTile: HyperRectangle,  # (N,Cin,Hx,Wx)
            dyFull: Tuple[int, int, int, int],  # full dY
            P: int,
            Q: int,
            pads: Tuple[int, int, int, int],  # (t,b,l,r)
            strides: Tuple[int, int],  # (sh, sw)
            dyC: int,  # Cout for this op
            dxAbsOff: Tuple[int, int, int, int],  # abs offset for boundary decision
    ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:

        (nOff, _cOff, _hxOff_rel, _wxOff_rel) = dxTile.offset
        (nSize, _cinSize, hxSize, wxSize) = dxTile.dims

        (_, _, hxOff_abs, wxOff_abs) = dxAbsOff

        pad_top, pad_bottom, pad_left, pad_right = pads
        sh, sw = strides

        hx0 = hxOff_abs
        hx1 = hxOff_abs + hxSize - 1
        wx0 = wxOff_abs
        wx1 = wxOff_abs + wxSize - 1

        Hy = dyFull[2]
        Wy = dyFull[3]

        oy0 = cls._ceil_div(hx0 - (P - 1) + pad_top, sh)
        oy1 = cls._floor_div(hx1 + pad_top, sh)
        ox0 = cls._ceil_div(wx0 - (Q - 1) + pad_left, sw)
        ox1 = cls._floor_div(wx1 + pad_left, sw)

        oy0 = max(0, oy0)
        ox0 = max(0, ox0)
        oy1 = min(Hy - 1, oy1)
        ox1 = min(Wy - 1, ox1)

        if oy0 > oy1 or ox0 > ox1:
            raise RuntimeError(
                f"dx tile {dxTile.offset}/{dxTile.dims} produces empty dy halo: "
                f"oy[{oy0},{oy1}] ox[{ox0},{ox1}] (Hy={Hy},Wy={Wy},P={P},Q={Q},pads={pads},strides={strides})")

        dyH = oy1 - oy0 + 1
        dyW = ox1 - ox0 + 1

        dyCube = HyperRectangle(
            (nOff, 0, oy0, ox0),  # dY: (N, C_out, H, W)
            (nSize, dyC, dyH, dyW))

        # tile-level ONNX pads only at boundary
        tile_pad_top = pad_top if oy0 == 0 else 0
        tile_pad_bottom = pad_bottom if (oy0 + dyH) == Hy else 0
        tile_pad_left = pad_left if ox0 == 0 else 0
        tile_pad_right = pad_right if (ox0 + dyW) == Wy else 0

        return dyCube, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)

    @staticmethod
    def _get_abs_off(abs_obj: AbsoluteHyperRectangle, fallback_rect: HyperRectangle):
        abs_off = getattr(abs_obj, "absoluteOffset", None)
        if abs_off is None:
            abs_off = getattr(abs_obj, "absolute_offset", None)
        if abs_off is None:
            abs_off = fallback_rect.offset
        return abs_off

    @classmethod
    def extraSerializeChecks(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> None:
        """Hook for DW checks etc."""
        return

    @classmethod
    def _make_weight_cube(cls, dxCube: HyperRectangle, wShape: Tuple[int, int, int, int]) -> HyperRectangle:
        """Per-tile weight cube. Regular conv: W layout [Cout, Cin/group, P, Q],
        so the Cin slice tracks dxCube.dims[1]. Subclasses override for DW."""
        return HyperRectangle(
            (0, dxCube.offset[1], 0, 0),
            (wShape[0], dxCube.dims[1], wShape[2], wShape[3]),
        )

    # ---------------------------------------------------
    # 4) serialize: dx tiles -> dy halo tiles
    # ---------------------------------------------------
    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        cls.extraSerializeChecks(ctxt, operatorRepresentation)

        varDY = operatorRepresentation[cls.gradOutKey]  # dY
        varW = operatorRepresentation[cls.weightKey]  # W
        varDX = operatorRepresentation[cls.gradInKey]  # dX

        _pads = list(operatorRepresentation.get("pads", [0, 0, 0, 0]))  # ONNX: [H_begin, W_begin, H_end, W_end]
        pads = (_pads[0], _pads[2], _pads[1], _pads[3])  # reorder to (top, bottom, left, right)
        strides = tuple(operatorRepresentation.get("strides", [1, 1]))  # (sh,sw)

        dyFull = tuple(ctxt.lookup(varDY).shape)  # (N,Cout,Ho,Wo)
        dxFull = tuple(ctxt.lookup(varDX).shape)  # (N,Cin,Hi,Wi)
        wShape = tuple(ctxt.lookup(varW).shape)  # (Cout,Cin/group,P,Q) or DW: (Cin,1,P,Q)

        P, Q = cls.get_kernel_hw(ctxt, varW, wShape)
        dyC = cls.get_dy_channels(ctxt, operatorRepresentation, varDY, varDX, varW, dyFull, dxFull, wShape)

        dxTiles = [c.rectangle for c in absoluteOutputCubes]

        # weight may be a Constant op excluded from the tiling solution
        varW_name = operatorRepresentation[cls.weightKey]
        weight_in_solution = varW_name in tilingSolution.tensorMemoryConstraints

        addrNames = [cls.gradOutKey]
        if weight_in_solution:
            addrNames.append(cls.weightKey)
        addrNames.append(cls.gradInKey)
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "ch_im_out": [],

            # unified template order:
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": [],
            "offset_grad_in_h": [],
            "offset_grad_in_w": [],
            "offset_grad_out_h": [],
            "offset_grad_out_w": [],
        }

        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
            "offset_grad_in_h": PointerClass(uint16_t),
            "offset_grad_in_w": PointerClass(uint16_t),
            "offset_grad_out_h": PointerClass(uint16_t),
            "offset_grad_out_w": PointerClass(uint16_t),
        }

        inputDyCubes: List[HyperRectangle] = []
        inputWCubes: List[HyperRectangle] = []
        outputDxCubes: List[HyperRectangle] = []

        ch_out = cls.get_ch_im_out(ctxt, dyFull, dxFull, wShape)

        for idx, dxCube in enumerate(dxTiles):
            abs_off = cls._get_abs_off(absoluteOutputCubes[idx], dxCube)

            dyCube, (tpt, tpb, tpl, tpr) = cls.computeDyCubeFromDxTile(
                dxTile = dxCube,
                dyFull = dyFull,
                P = P,
                Q = Q,
                pads = pads,
                strides = strides,
                dyC = dyC,  # IMPORTANT: use computed dyC
                dxAbsOff = abs_off)

            # Per-tile W cube: layout differs between regular and DW conv,
            # so delegate to the subclass-overridable helper.
            wCube = cls._make_weight_cube(dxCube, wShape)

            replacements["dim_im_in_x"].append(dxCube.dims[2])  # H_in_tile
            replacements["dim_im_in_y"].append(dxCube.dims[3])  # W_in_tile
            replacements["dim_im_out_x"].append(dyCube.dims[2])  # H_out_tile (halo)
            replacements["dim_im_out_y"].append(dyCube.dims[3])  # W_out_tile (halo)

            replacements["ch_im_in"].append(dxCube.dims[1])
            replacements["ch_im_out"].append(ch_out)

            # ConvGradX kernels compute `base = ox*sw - pad_left` (abs coord)
            # to find the dX cell each dY pixel scatters into; this needs the
            # *original* (global) ONNX pad, NOT the tile-boundary-adjusted
            # pad. Using the tile-adjusted pad for a non-boundary tile makes
            # `base` off by the full-op pad amount → off-by-N shift in the
            # written dX columns (visible as dX values rotated by one for
            # DW stride=2 pad=1 when HW is tiled; see ConvGradX_DW_block_1_s2).
            # The per-tile adjustment is still passed via kx/ky pruning (the
            # kernel bounds-checks each written cell against the tile extents
            # hx0/hx1/wx0/wx1) so non-boundary tiles correctly drop kernel
            # positions that would land outside their own region.
            _py_top, _py_bottom, _px_left, _px_right = cls.map_onnx_pads_to_template(pads[0], pads[1], pads[2], pads[3])
            replacements["padding_y_top"].append(_py_top)
            replacements["padding_y_bottom"].append(_py_bottom)
            replacements["padding_x_left"].append(_px_left)
            replacements["padding_x_right"].append(_px_right)

            replacements["offset_grad_in_h"].append(abs_off[2])
            replacements["offset_grad_in_w"].append(abs_off[3])
            replacements["offset_grad_out_h"].append(dyCube.offset[2])
            replacements["offset_grad_out_w"].append(dyCube.offset[3])

            inputDyCubes.append(dyCube)
            inputWCubes.append(wCube)
            outputDxCubes.append(dxCube)

        if weight_in_solution:
            inputLoadSchedule = [{cls.gradOutKey: dy, cls.weightKey: w} for dy, w in zip(inputDyCubes, inputWCubes)]
        else:
            inputLoadSchedule = [{cls.gradOutKey: dy} for dy in inputDyCubes]
        outputLoadSchedule = [{cls.gradInKey: dx} for dx in outputDxCubes]

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        # ── Post-solve roofline cost estimation for ConvGradX ──
        try:
            import math
            num_tiles = len(absoluteOutputCubes)
            bpe = 4
            peak = _HW_PARAMS['peak_flops_per_cycle']
            setup = _HW_PARAMS['dma_setup_cycles']

            # Use first tile's shapes as representative
            dx_tile_shape = absoluteOutputCubes[0].rectangle.dims  # (N, Cin_t, Hi_t, Wi_t)
            dy_tile_shape = inputDyCubes[0].dims if inputDyCubes else dyFull
            w_bytes = math.prod(wShape) * bpe
            dx_tile_bytes = math.prod(dx_tile_shape) * bpe
            dy_tile_bytes = math.prod(dy_tile_shape) * bpe

            # ConvGradX FLOPs per tile: 2 × N × Cout × Cin_tile × kH × kW × Hi_tile × Wi_tile
            N_t = dx_tile_shape[0] if len(dx_tile_shape) > 0 else 1
            Ci_t = dx_tile_shape[1] if len(dx_tile_shape) > 1 else 1
            Hi_t = dx_tile_shape[2] if len(dx_tile_shape) > 2 else 1
            Wi_t = dx_tile_shape[3] if len(dx_tile_shape) > 3 else 1
            Cout = dyFull[1]
            tile_flops = 2.0 * N_t * Cout * Ci_t * P * Q * Hi_t * Wi_t

            # W is typically full (not tiled) → hoistable if same every tile
            w_bw = _tensor_bw(ctxt, varW)
            dy_bw = _tensor_bw(ctxt, varDY)
            dx_bw = _tensor_bw(ctxt, varDX)

            # W hoistable if not tiled (same across all tiles)
            w_hoistable = not weight_in_solution or (len(inputWCubes) > 1 and all(
                c.offset == inputWCubes[0].offset and c.dims == inputWCubes[0].dims
                for c in inputWCubes[1:])) if inputWCubes else True

            # Per-tile DMA (SB serial)
            tile_dma = dy_tile_bytes / dy_bw + setup + dx_tile_bytes / dx_bw + setup
            if not w_hoistable:
                tile_dma += w_bytes / w_bw + setup
            hoisted_dma = 0.0
            if w_hoistable:
                hoisted_dma = w_bytes / w_bw + setup

            # Per-tile bytes for AI calculation (data that moves per tile)
            tile_bytes = dy_tile_bytes + dx_tile_bytes
            if not w_hoistable:
                tile_bytes += w_bytes

            tile_AI = tile_flops / tile_bytes if tile_bytes > 0 else float('inf')
            min_bw = min(dy_bw, dx_bw)
            if not w_hoistable:
                min_bw = min(min_bw, w_bw)
            tile_attainable = min(peak, tile_AI * min_bw)
            tile_compute = tile_flops / tile_attainable if tile_attainable > 0 else 0

            total_cycles = hoisted_dma + num_tiles * (tile_dma + tile_compute)
            total_flops = 2.0 * dxFull[0] * Cout * dxFull[1] * P * Q * dxFull[2] * dxFull[3]
            ideal = total_flops / peak

            node = operatorRepresentation.get('nodeName', '?')
            w_level = 'L2' if w_bw == _HW_PARAMS['bw_l2_to_l1'] else 'L3'
            dy_level = 'L2' if dy_bw == _HW_PARAMS['bw_l2_to_l1'] else 'L3'
            # print(f"[TileCost:GradX] {node}: tiles={num_tiles} "
            #       f"AI={tile_AI:.2f} attainable={tile_attainable:.1f}F/cyc "
            #       f"({'compute' if tile_compute >= tile_dma else 'memory'}-bound) "
            #       f"overhead={total_cycles/ideal:.2f}x "
            #       f"tile_compute={tile_compute:.0f} tile_dma={tile_dma:.0f} "
            #       f"total={total_cycles:.0f} ideal={ideal:.0f} "
            #       f"dx_tile={list(dx_tile_shape)} dy_tile={list(dy_tile_shape)} "
            #       f"W={list(wShape)}({w_level},{'hoist' if w_hoistable else 'per-tile'}) "
            #       f"dY({dy_level})")
        except Exception:
            pass

        return variableReplacementSchedule, tilingSchedule


# ============================================================================
# ConvGradX: subclass reusing the base
# ============================================================================


class ConvGradX2DHWTileConstraint(ConvGradXTileConstraintBase):
    pass


class ConvGradX2DIm2ColHWTileConstraint(ConvGradXTileConstraintBase):
    pass


class PWConvGradXTileConstraint(ConvGradXTileConstraintBase):
    """Pointwise (1x1) ConvGradX policy: pin HW=full, let the tiler split Cin.

    For PW layers HW is the only "spatial" axis but it doesn't carry kernel
    halo (kernel 1x1, stride 1, pad 0). The direct axpy kernel runs over
    HW as its innermost loop, so a small HW tile (<16 elements) blows the
    overhead-to-useful-work ratio past 50%. The tiler's default cost model
    will happily split HW into single-pixel tiles -- this used to produce
    the catastrophic 18- and 12-tile schedules on MobileNetV1 block_11/12
    (Cin=Cout=256, NHW=9), where 95% of cycles went into per-tile DMA and
    sync overhead instead of compute. Pinning HW full forces Cin to absorb
    all of the tiling pressure; with Cin tiled the per-tile compute is a
    full HW reduction that the inner loop amortises well.
    """

    # Only pin HW=full when the resulting dY full fits comfortably in L1
    # alongside a Cin tile + dX tile. Early MobileNetV1 PW layers (e.g.
    # block_0 with NHW=2304, dY full = 144 KB) violate this budget; for
    # those we leave HW free since the direct axpy kernel handles a long
    # HW inner loop efficiently anyway.
    HW_PIN_BUDGET_BYTES = 24 * 1024

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        super().addPolicyConstraint(tilerModel, parseDict, ctxt)

        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]

        dyBuf = ctxt.lookup(dyName)
        dxBuf = ctxt.lookup(dxName)

        # Estimate dY full byte size (fp32 assumed; PW is single-precision in
        # this stack). Skip pinning when full dY exceeds the HW-pin budget.
        N, Cout, H_y, W_y = dyBuf.shape
        bytes_per_elem = dyBuf._type.referencedType.typeWidth // 8
        dy_full_bytes = N * Cout * H_y * W_y * bytes_per_elem
        if dy_full_bytes > cls.HW_PIN_BUDGET_BYTES:
            return tilerModel

        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 2) == dyBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 3) == dyBuf.shape[3])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 2) == dxBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 3) == dxBuf.shape[3])

        return tilerModel


class DWConvGradX2DTileConstraint(ConvGradXTileConstraintBase):
    """
    Depthwise ConvGradX (dX) tiling, reusing ConvGradXTileConstraintBase.

    Expected tensors:
      data_in  = grad_out (dY) [N, C, H_out, W_out]
      data_out = grad_in  (dX) [N, C, H_in,  W_in]
      weight   = W        [C, 1, P, Q]
    """

    # If your DW parser uses different keys, override here.
    gradOutKey = "grad_out"
    gradInKey = "grad_in"
    weightKey = "weight"

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dyName = parseDict[cls.gradOutKey]  # dY
        dxName = parseDict[cls.gradInKey]  # dX
        wName = parseDict[cls.weightKey]  # W

        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dxName)
        tilerModel.addTensorDimToModel(ctxt, wName)

        # N match
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 0) == tilerModel.getTensorDimVar(dxName, 0))

        # DW channels: Cin == Cout == W[0], and W[1]==1
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == tilerModel.getTensorDimVar(wName, 0))
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 1) == tilerModel.getTensorDimVar(wName, 0))
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 1) == 1)

        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        DW policy:
          - keep full channels (C)
          - weight not tiled
          - enforce W[1]==1
        """
        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]
        wName = parseDict[cls.weightKey]

        dyBuf = ctxt.lookup(dyName)
        dxBuf = ctxt.lookup(dxName)
        wBuf = ctxt.lookup(wName)

        # Full channels for both dY and dX
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == dyBuf.shape[1])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 1) == dxBuf.shape[1])

        # Weight not tiled + DW second dim fixed to 1
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 0) == wBuf.shape[0])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 1) == 1)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 2) == wBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 3) == wBuf.shape[3])

        return tilerModel

    @classmethod
    def extraSerializeChecks(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> None:
        varDY = operatorRepresentation[cls.gradOutKey]
        varDX = operatorRepresentation[cls.gradInKey]
        varW = operatorRepresentation[cls.weightKey]

        dyFull = tuple(ctxt.lookup(varDY).shape)  # (N,C,Ho,Wo)
        dxFull = tuple(ctxt.lookup(varDX).shape)  # (N,C,Hi,Wi)
        wShape = tuple(ctxt.lookup(varW).shape)  # (C,1,P,Q)

        Cin = dxFull[1]
        Cout = dyFull[1]
        if Cin != Cout:
            raise RuntimeError(f"DWConvGradX expects Cin==Cout, got Cin={Cin}, Cout={Cout}")
        if wShape[0] != Cin:
            raise RuntimeError(f"DWConvGradX expects W[0]==C, got W[0]={wShape[0]} vs C={Cin}")
        if wShape[1] != 1:
            raise RuntimeError(f"DWConvGradX expects W[1]==1, got {wShape[1]}")

    @classmethod
    def get_dy_channels(
        cls,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
        dyName: str,
        dxName: str,
        wName: str,
        dyFull: Tuple[int, int, int, int],
        dxFull: Tuple[int, int, int, int],
        wShape: Tuple[int, int, int, int],
    ) -> int:
        # DW: dY channels is C
        return dyFull[1]

    @classmethod
    def _make_weight_cube(cls, dxCube: HyperRectangle, wShape: Tuple[int, int, int, int]) -> HyperRectangle:
        # DW weight layout [C, 1, P, Q]: the Cout axis (dim 0) tracks dxCube's
        # channel axis, Cin is always 1. Returning the regular-conv cube here
        # multiplies the weight transfer by Cin and blows past L1.
        return HyperRectangle(
            (dxCube.offset[1], 0, 0, 0),
            (dxCube.dims[1], 1, wShape[2], wShape[3]),
        )


# =================================================================================
# ConvGradW tiling strategies
#
# Each strategy encapsulates a tiling regime for the ConvGradW family:
#   - applies(): when this strategy is feasible / preferred for a given layer
#   - add_constraints(): policy constraints emitted to the OR-tools tiler model
#   - matches_solution(): recognize this strategy's shape in a tiler-returned solution
#   - serialize(): emit the per-tile schedule for codegen
#
# The TileConstraint base class dispatches addPolicyConstraint / serializeTilingSolution
# through its `strategies` class attribute (ordered priority list).
# =================================================================================


def _tensor_bytes(buf) -> int:
    """Total byte size of a Deeploy buffer."""
    n = 1
    for d in buf.shape:
        n *= d
    return n * (buf._type.referencedType.typeWidth // 8)


# Empirical: dY budget so that CinSlice fits with dW + X scratch in 128KB L1.
# Layers with dy_bytes <= 32KB use CinSlice; otherwise spatial tiling kicks in.
L1_DY_BUDGET_BYTES = 32 * 1024

# ============================================================================
# Roofline cost model — hardware parameters from gvsoc Siracusa config
# ============================================================================
# Sources: AI_AGENT/GVSOC/siracusa_bandwidth_parameters.md
#   L2→L1: MCHAN 4 ports × 4B = 16 B/cycle @ 50 MHz cluster clock
#   L3→L2: HyperBus 1 byte/periph_clock_cycle @ 50 MHz = 1 B/cycle
#   L3→L1: L3→L2(1 B/cycle)→L1(16 B/cycle), bottleneck = L3→L2 = 1 B/cycle
_HW_PARAMS = {
    'peak_flops_per_cycle': 16.0,  # 8 cores × 1 FMA/cycle (FP32)
    'bw_l2_to_l1': 16.0,  # MCHAN: 4 local ports × 4B width
    'bw_l3_to_l1': 1.0,  # HyperBus: 1 B/periph_cycle @ 50 MHz
    'dma_setup_cycles': 200,  # per-transaction fixed cost (empirical)
    'l1_size': 128 * 1024,  # default L1 TCDM
    'cluster_ico_latency': 2,  # cluster interconnect latency (cycles)
}


def _tensor_bw(ctxt, name):
    """Bandwidth (B/cycle) based on tensor memory level."""
    try:
        return _HW_PARAMS['bw_l2_to_l1'] if getattr(ctxt.lookup(name), '_memoryLevel',
                                                    'L3') == 'L2' else _HW_PARAMS['bw_l3_to_l1']
    except Exception:
        return _HW_PARAMS['bw_l3_to_l1']


def _post_solve_cost(strategy_name, num_tiles, tile_tensors, tile_shapes, ctxt, parseDict, owner_cls):
    """Post-solve roofline cost estimation using ACTUAL tile sizes from solver.

    Per-tile roofline model:
      tile_AI = tile_FLOPs / tile_bytes_moved
      tile_attainable_perf = min(peak, tile_AI × bandwidth)
      tile_compute = tile_FLOPs / tile_attainable_perf

    SB (single buffer) model: DMA and compute are serial per tile.
      tile_cost = tile_dma + tile_compute
      total = hoisted_dma + num_tiles × tile_cost

    Args:
        tile_shapes: dict {tensor_key: tile_shape_tuple} from solver
    """
    peak = _HW_PARAMS['peak_flops_per_cycle']
    setup = _HW_PARAMS['dma_setup_cycles']

    # Tile-level FLOPs from actual tile shapes
    dy_tile = tile_shapes.get(owner_cls.gradOutKey, ())
    x_tile = tile_shapes.get(owner_cls.dataInKey, ())
    dw_tile = tile_shapes.get(owner_cls.weightKey, ())

    dwBuf = ctxt.lookup(parseDict[owner_cls.weightKey])
    kH, kW = dwBuf.shape[2], dwBuf.shape[3]

    if strategy_name == "cin_slice":
        # ConvGradW CinSlice: reduction over Ho×Wo, tile along Cin
        # tile FLOPs = 2 × N × Cout × Cin_tile × kH × kW × Ho × Wo
        N_t = dy_tile[0] if len(dy_tile) > 0 else 1
        Co_t = dy_tile[1] if len(dy_tile) > 1 else 1
        Ho_t = dy_tile[2] if len(dy_tile) > 2 else 1
        Wo_t = dy_tile[3] if len(dy_tile) > 3 else 1
        Ci_t = dw_tile[1] if len(dw_tile) > 1 else 1
        tile_flops = 2.0 * N_t * Co_t * Ci_t * kH * kW * Ho_t * Wo_t
    elif strategy_name == "cout_hw_slice":
        # CoutHWSlice: tile Cout and/or HW on dY
        N_t = dy_tile[0] if len(dy_tile) > 0 else 1
        Co_t = dy_tile[1] if len(dy_tile) > 1 else 1
        Ho_t = dy_tile[2] if len(dy_tile) > 2 else 1
        Wo_t = dy_tile[3] if len(dy_tile) > 3 else 1
        Ci_t = x_tile[1] if len(x_tile) > 1 else 1
        tile_flops = 2.0 * N_t * Co_t * Ci_t * kH * kW * Ho_t * Wo_t
    else:
        tile_flops = 0.0

    # Per-tile DMA bytes and bandwidth
    tile_dma_cycles = 0.0
    hoisted_dma_cycles = 0.0
    tile_bytes_total = 0.0
    breakdown = []

    for tname, tbytes, hoistable in tile_tensors:
        bw = _tensor_bw(ctxt, parseDict.get(tname, tname))
        level = 'L2' if bw == _HW_PARAMS['bw_l2_to_l1'] else 'L3'
        xfer_cost = tbytes / bw + setup
        if hoistable:
            hoisted_dma_cycles += xfer_cost
            breakdown.append(f"  {tname}({level}): {tbytes}B hoisted = {xfer_cost:.0f}cyc")
        else:
            tile_dma_cycles += xfer_cost
            tile_bytes_total += tbytes
            breakdown.append(f"  {tname}({level}): {tbytes}B/tile × {num_tiles} = {xfer_cost * num_tiles:.0f}cyc")

    # Roofline: per-tile arithmetic intensity and attainable performance
    if tile_bytes_total > 0:
        tile_AI = tile_flops / tile_bytes_total
    else:
        tile_AI = float('inf')

    # Effective bandwidth = weighted average of per-tensor bandwidths
    # (simplified: use min bandwidth among per-tile tensors as bottleneck)
    min_bw = _HW_PARAMS['bw_l3_to_l1']
    for tname, tbytes, hoistable in tile_tensors:
        if not hoistable:
            bw = _tensor_bw(ctxt, parseDict.get(tname, tname))
            min_bw = min(min_bw, bw) if tbytes > 0 else min_bw

    tile_attainable_perf = min(peak, tile_AI * min_bw)
    tile_compute_cycles = tile_flops / tile_attainable_perf if tile_attainable_perf > 0 else 0

    # SB model: DMA and compute are serial per tile
    tile_total = tile_dma_cycles + tile_compute_cycles
    total_cycles = hoisted_dma_cycles + num_tiles * tile_total

    # Full-layer theoretical (no tiling overhead, peak utilization)
    dyBuf = ctxt.lookup(parseDict[owner_cls.gradOutKey])
    xBuf = ctxt.lookup(parseDict[owner_cls.dataInKey])
    N, Co, Ho, Wo = dyBuf.shape
    _, Ci, Hi, Wi = xBuf.shape
    total_flops = 2.0 * N * Co * Ci * kH * kW * Ho * Wo
    ideal_cycles = total_flops / peak

    return {
        'tile_flops': tile_flops,
        'tile_bytes': tile_bytes_total,
        'tile_AI': tile_AI,
        'tile_attainable_perf': tile_attainable_perf,
        'tile_compute_cycles': tile_compute_cycles,
        'tile_dma_cycles': tile_dma_cycles,
        'hoisted_dma_cycles': hoisted_dma_cycles,
        'total_cycles': total_cycles,
        'ideal_cycles': ideal_cycles,
        'overhead': total_cycles / ideal_cycles if ideal_cycles > 0 else float('inf'),
        'bound': 'compute' if tile_compute_cycles >= tile_dma_cycles else 'memory',
        'num_tiles': num_tiles,
        'strategy': strategy_name,
        'breakdown': breakdown,
    }


class GradWStrategy:
    """Abstract base for a ConvGradW tiling strategy."""
    name: str = "abstract"

    @classmethod
    def applies(cls, owner_cls, ctxt: NetworkContext, parseDict: Dict) -> bool:
        raise NotImplementedError

    @classmethod
    def add_constraints(cls, owner_cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        raise NotImplementedError

    @classmethod
    def matches_solution(cls, owner_cls, tilingSolution: NodeMemoryConstraint,
                         absoluteOutputCubes: List[AbsoluteHyperRectangle], targetMemLevel: str, ctxt: NetworkContext,
                         operatorRepresentation: OperatorRepresentation) -> bool:
        raise NotImplementedError

    @classmethod
    def serialize(cls, owner_cls, tilingSolution: NodeMemoryConstraint,
                  absoluteOutputCubes: List[AbsoluteHyperRectangle], targetMemLevel: str, ctxt: NetworkContext,
                  operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        raise NotImplementedError


class CinSliceStrategy(GradWStrategy):
    """Tile dW along Cin, keep dY full. GEMM K = Hout*Wout.

    Feasible when dY fits in L1 (dy_bytes <= L1_DY_BUDGET_BYTES). Each Cin
    slice of dW is independent: dW[:, ci, :, :] only needs X[:, ci, :, :]
    and the full dY. The kernel accumulates partial Cin contributions via
    mm_add across tiles.

    Best for: small spatial / large channel layers (e.g. ResNet8 layer3).
    """
    name = "cin_slice"

    @classmethod
    def applies(cls, owner_cls, ctxt, parseDict):
        dyBuf = ctxt.lookup(parseDict[owner_cls.gradOutKey])
        return _tensor_bytes(dyBuf) <= L1_DY_BUDGET_BYTES

    @classmethod
    def add_constraints(cls, owner_cls, tilerModel, parseDict, ctxt):
        xName = parseDict[owner_cls.dataInKey]
        dyName = parseDict[owner_cls.gradOutKey]
        dwName = parseDict[owner_cls.weightKey]

        xBuf = ctxt.lookup(xName)
        dyBuf = ctxt.lookup(dyName)
        dwBuf = ctxt.lookup(dwName)

        # Cout full on dY (keeps K = Hout*Wout large for GEMM)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == dyBuf.shape[1])
        # dY spatial full (no HW tiling)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 2) == dyBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 3) == dyBuf.shape[3])

        # dW: Cout (dim 0) full, kH/kW full; Cin (dim 1) allowed to tile
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 0) == dwBuf.shape[0])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 2) == dwBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 3) == dwBuf.shape[3])

        # X: Cin (dim 1) tiles in lockstep with dW dim 1 (via geometrical constraint).
        # Spatial full on X (no HW tiling since dY spatial is full)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 2) == xBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 3) == xBuf.shape[3])

        return tilerModel

    @classmethod
    def matches_solution(cls, owner_cls, tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt,
                         operatorRepresentation):
        # CinSlice keeps dY spatial full. Use dY tile shape from tilingSolution.
        dyName = operatorRepresentation[owner_cls.gradOutKey]
        dyFull = tuple(ctxt.lookup(dyName).shape)
        try:
            dyShape = tilingSolution.tensorMemoryConstraints[dyName].memoryConstraints[targetMemLevel].shape
        except (KeyError, AttributeError):
            return True  # no tiling info → assume default (CinSlice)
        return dyShape[2] == dyFull[2] and dyShape[3] == dyFull[3]

    @classmethod
    def serialize(cls, owner_cls, tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation):
        owner_cls.extraSerializeChecks(ctxt, operatorRepresentation)

        xName = operatorRepresentation[owner_cls.dataInKey]
        dyName = operatorRepresentation[owner_cls.gradOutKey]
        dwName = operatorRepresentation[owner_cls.weightKey]

        _pads = list(operatorRepresentation.get("pads", [0, 0, 0, 0]))  # ONNX: [H_begin, W_begin, H_end, W_end]
        pads = (_pads[0], _pads[2], _pads[1], _pads[3])  # (top, bottom, left, right)

        xFull = tuple(ctxt.lookup(xName).shape)  # (N, Cin, Hi, Wi)
        dyFull = tuple(ctxt.lookup(dyName).shape)  # (N, Cout, Ho, Wo)
        dwShape = tuple(ctxt.lookup(dwName).shape)  # (Cout, Cin_per_group, P, Q)

        Cin_full = xFull[1]
        Cout_full = dyFull[1]
        N_tile = dyFull[0]
        pad_top, pad_bottom, pad_left, pad_right = pads

        addrNames = [owner_cls.dataInKey, owner_cls.gradOutKey, owner_cls.weightKey]
        inputBaseOffsets, outputBaseOffsets = owner_cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                        operatorRepresentation, addrNames)

        replacements: Dict[str, List[int]] = {
            k: [] for k in [
                "dim_im_in_x",
                "dim_im_in_y",
                "dim_im_out_x",
                "dim_im_out_y",
                "ch_im_in",
                "ch_im_out",
                "padding_y_top",
                "padding_y_bottom",
                "padding_x_left",
                "padding_x_right",
            ]
        }
        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
        }

        # Derive Cin slices from dW output cubes (cubes are dW slabs).
        ci_slices: List[Tuple[int, int]] = []
        for cube in absoluteOutputCubes:
            abs_off = getattr(cube, 'absoluteOffset', None)
            if abs_off is None:
                abs_off = cube.rectangle.offset
            ciOff = abs_off[1]
            ciSz = cube.rectangle.dims[1]
            ci_slices.append((ciOff, ciSz))
        if not ci_slices:
            ci_slices.append((0, Cin_full))

        inputLoadSchedule = []
        outputLoadSchedule = []
        for ciOff, ciSz in ci_slices:
            dwTile = HyperRectangle((0, ciOff, 0, 0), (dwShape[0], ciSz, dwShape[2], dwShape[3]))
            dyTile = HyperRectangle((0, 0, 0, 0), (N_tile, Cout_full, dyFull[2], dyFull[3]))
            xTile = HyperRectangle((0, ciOff, 0, 0), (xFull[0], ciSz, xFull[2], xFull[3]))

            replacements["dim_im_in_x"].append(xFull[2])
            replacements["dim_im_in_y"].append(xFull[3])
            replacements["dim_im_out_x"].append(dyFull[2])
            replacements["dim_im_out_y"].append(dyFull[3])
            replacements["ch_im_in"].append(ciSz)
            replacements["ch_im_out"].append(Cout_full)
            replacements["padding_y_top"].append(pad_top)
            replacements["padding_y_bottom"].append(pad_bottom)
            replacements["padding_x_left"].append(pad_left)
            replacements["padding_x_right"].append(pad_right)

            inputLoadSchedule.append({owner_cls.dataInKey: xTile, owner_cls.gradOutKey: dyTile})
            outputLoadSchedule.append({owner_cls.weightKey: dwTile})

        return (VariableReplacementScheme(replacements, replacementTypes),
                TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule))


class CoutHWSliceStrategy(GradWStrategy):
    """Free Cout / HW tiling on dY; X Cin and dW Cin/kernel kept full.

    Lets the OR-tools tiler split between Cout slicing and HW slicing per
    layer. dW slices are along Cout (disjoint per Cout slab); each (ho, wo)
    tile within a Cout slab accumulates partial dW via mm_add. GEMM K
    dimension degrades to tile_H * tile_W when HW tiling is needed.

    Used when dY doesn't fit comfortably in L1 with full Cout (e.g.
    MobileNetV1 stem: dY = 16x96x96 = 576KB > L1). Equivalent to the
    pre-Cin-slice devel default policy.

    Constraints:
      - X Cin (dim 1) full
      - dW Cin (dim 1) / kH / kW full; Cout (dim 0) free
      - dY HW free (>= 1); dY Cout free
    """
    name = "cout_hw_slice"

    @classmethod
    def applies(cls, owner_cls, ctxt, parseDict):
        # Always-applies fallback. Caller checks CinSlice first.
        return True

    @classmethod
    def add_constraints(cls, owner_cls, tilerModel, parseDict, ctxt):
        xName = parseDict[owner_cls.dataInKey]
        dyName = parseDict[owner_cls.gradOutKey]
        dwName = parseDict[owner_cls.weightKey]

        xBuf = ctxt.lookup(xName)
        dwBuf = ctxt.lookup(dwName)

        # Full Cin on X (reduction axis for dW is spatial; Cin is independent per Cout slice)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 1) == xBuf.shape[1])

        # dW: keep Cin / kH / kW full; allow Cout (dim 0) to tile
        for d in range(1, len(dwBuf.shape)):
            tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, d) == dwBuf.shape[d])

        # ── Forbid simultaneous Cout + H/W tiling for regular ConvGradW ──
        # dW of a Cout slab is a spatial REDUCTION accumulated across that slab's
        # H/W tiles. The template zeroes dW once per slab and lets the H/W tiles
        # add their partials. When BOTH Cout and H/W tile, the per-slab dW does
        # NOT reliably persist/accumulate across the slab's H/W tiles under the
        # single-buffer schedule, so only the last H/W partial survives → wrong
        # dW → loss drift (Siracusa ResNet8 layer1: dY 16×32×32 = 64KB > CinSlice
        # budget → falls here; the spatial-link otherwise lets the solver tile
        # Cout *and* W). Forcing Cout full keeps a single slab (the validated
        # memset-once-then-accumulate path) while the spatial-link still tiles
        # H/W into a few large strips (so GAP9's large-stem feasibility holds).
        # DW/PW ConvGradW genuinely need Cout tiling (per-channel layout) and do
        # not hit this combination in practice, so they opt out via the flag.
        # The drift comes from tiling Cout *and* H/W simultaneously. Two ways to
        # forbid that combo:
        #   (a) Cout full, H/W tiled  -> single dW slab, H/W partials accumulate;
        #   (b) H/W full, Cout tiled  -> each Cout slab computes its dW in one
        #       spatial pass (no cross-H/W accumulation, so no drift either).
        # (a) is the validated default. But it pins the *entire* dW in L1, which is
        # infeasible for large-Cout convs (e.g. TSDR patch-embed: dW=128x80x3 ~123KB
        # > L1). For those, fall back to (b): force H/W full and let Cout tile. Small
        # dW keeps path (a) so existing Siracusa/GAP9 numerics are unchanged.
        if getattr(owner_cls, "coutHWSlice_force_cout_full", False):
            dyBuf = ctxt.lookup(dyName)
            dwElems = 1
            for _d in dwBuf.shape:
                dwElems *= int(_d)
            dwBytes = dwElems * 4  # float32 ConvGradW
            COUT_FULL_MAX_DW_BYTES = 65536
            if dwBytes <= COUT_FULL_MAX_DW_BYTES:
                tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == dwBuf.shape[0])
                tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 0) == dwBuf.shape[0])
            # else: dW too large to pin Cout-full in L1 (e.g. TSDR patch-embed). Add
            # no Cout/HW restriction and let the solver tile freely — feasibility wins;
            # any dW-accumulation drift is caught by the numerical reference check.
            # Siracusa models have no large-dW regular convs, so their path is the
            # Cout-full branch above and stays numerically unchanged.

        # dY tile spatial dims >= 1 (tiler picks)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 2) >= 1)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 3) >= 1)

        # ── X ↔ dY spatial link (boundary-aware forward-conv relation) ──
        # Without this the model has NO relation between the X (input) spatial
        # tile and the dY (output-gradient) spatial tile — only channel/batch
        # are linked in addGeometricalConstraint. The solver is then free to
        # pick a degenerate 1×1 dY tile with an independently-sized (tiny) X
        # tile; for large-spatial regular ConvGradW (e.g. MobileNetV1 stem,
        # dY 8×48×48, X 3×96×96) this yields a per-element schedule (48·48·8 =
        # 18432 tiles) whose per-tile PI_L1 metadata (~190KB) overflows L1 at
        # link time. Mirroring Conv2DTileConstraintCHW's geometrical relation
        # ties the X tile to exactly the dY tile's input halo (with the
        # full-tile padding indicator), so the model accounts for the real X
        # cost and `random-max` maximizes dY (X grows in lockstep, filling L1)
        # → a few large H/W strips. CinSlice keeps spatial full so the relation
        # holds trivially there; this is added only on the HW-slicing path.
        try:
            pads = parseDict["pads"]
            strides = parseDict["strides"]
            dil = parseDict.get("dilations", [1, 1])
            kH, kW = dwBuf.shape[2], dwBuf.shape[3]
            fullXH, fullXW = xBuf.shape[2], xBuf.shape[3]
            xH = tilerModel.getTensorDimVar(xName, 2)
            xW = tilerModel.getTensorDimVar(xName, 3)
            dyH = tilerModel.getTensorDimVar(dyName, 2)
            dyW = tilerModel.getTensorDimVar(dyName, 3)
            effXH = xH + ((pads[0] + pads[2]) * (xH == fullXH)) - ((kH - 1) * (xH != fullXH))
            effXW = xW + ((pads[1] + pads[3]) * (xW == fullXW)) - ((kW - 1) * (xW != fullXW))
            tilerModel.addConstraint(dyH == (effXH - dil[0] * (kH - 1) - 1) // strides[0] + 1)
            tilerModel.addConstraint(dyW == (effXW - dil[1] * (kW - 1) - 1) // strides[1] + 1)
        except Exception:
            pass  # best-effort; never block tiling on the spatial-link cost model

        return tilerModel

    @classmethod
    def matches_solution(cls, owner_cls, tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt,
                         operatorRepresentation):
        # Matches when EITHER dy spatial < full OR dw Cout < full.
        dyName = operatorRepresentation[owner_cls.gradOutKey]
        dwName = operatorRepresentation[owner_cls.weightKey]
        dyFull = tuple(ctxt.lookup(dyName).shape)
        dwFull = tuple(ctxt.lookup(dwName).shape)
        try:
            dyShape = tilingSolution.tensorMemoryConstraints[dyName].memoryConstraints[targetMemLevel].shape
            dwShape = tilingSolution.tensorMemoryConstraints[dwName].memoryConstraints[targetMemLevel].shape
        except (KeyError, AttributeError):
            return False
        return (dyShape[2] < dyFull[2] or dyShape[3] < dyFull[3] or dwShape[0] < dwFull[0])

    @classmethod
    def serialize(cls, owner_cls, tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation):
        owner_cls.extraSerializeChecks(ctxt, operatorRepresentation)

        xName = operatorRepresentation[owner_cls.dataInKey]
        dyName = operatorRepresentation[owner_cls.gradOutKey]
        dwName = operatorRepresentation[owner_cls.weightKey]

        _pads = list(operatorRepresentation.get("pads", [0, 0, 0, 0]))
        pads = (_pads[0], _pads[2], _pads[1], _pads[3])  # (top, bottom, left, right)
        strides = tuple(operatorRepresentation.get("strides", [1, 1]))

        xFull = tuple(ctxt.lookup(xName).shape)
        dyFull = tuple(ctxt.lookup(dyName).shape)
        dwShape = tuple(ctxt.lookup(dwName).shape)

        # Tiler-picked dY tile shape at this mem level (fall back to full when missing)
        try:
            dyTileShape = tilingSolution.tensorMemoryConstraints[dyName].memoryConstraints[targetMemLevel].shape
        except Exception:
            dyTileShape = dyFull

        N_tile = dyTileShape[0]
        Ho_tile_max = dyTileShape[2]
        Wo_tile_max = dyTileShape[3]

        # Generate (ho, wo) tile grid covering full dY spatial extent
        Ho_full = dyFull[2]
        Wo_full = dyFull[3]

        h_tiles: List[Tuple[int, int]] = []
        w_tiles: List[Tuple[int, int]] = []

        ho = 0
        while ho < Ho_full:
            hs = min(Ho_tile_max, Ho_full - ho)
            h_tiles.append((ho, hs))
            ho += hs

        wo = 0
        while wo < Wo_full:
            ws = min(Wo_tile_max, Wo_full - wo)
            w_tiles.append((wo, ws))
            wo += ws

        addrNames = [owner_cls.dataInKey, owner_cls.gradOutKey, owner_cls.weightKey]
        inputBaseOffsets, outputBaseOffsets = owner_cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                        operatorRepresentation, addrNames)

        replacements: Dict[str, List[int]] = {
            k: [] for k in [
                "dim_im_in_x",
                "dim_im_in_y",
                "dim_im_out_x",
                "dim_im_out_y",
                "ch_im_in",
                "ch_im_out",
                "padding_y_top",
                "padding_y_bottom",
                "padding_x_left",
                "padding_x_right",
            ]
        }
        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
        }

        Cin_full = xFull[1]
        Cout_full = dyFull[1]

        # Cout tile size from tiler solution (falls back to full when not tiled)
        try:
            dwTileShape = tilingSolution.tensorMemoryConstraints[dwName].memoryConstraints[targetMemLevel].shape
            Cout_tile_max = dwTileShape[0]
        except Exception:
            Cout_tile_max = Cout_full

        # Derive Cout slices from absoluteOutputCubes (each cube is a dW Cout slab
        # at L3; the L1 schedule iterates per cube).
        co_slices: List[Tuple[int, int]] = []
        for cube in absoluteOutputCubes:
            coOff = cube.absoluteOffset[0]
            coSz = cube.rectangle.dims[0]
            co_slices.append((coOff, coSz))
        if not co_slices:
            co = 0
            while co < Cout_full:
                cs = min(Cout_tile_max, Cout_full - co)
                co_slices.append((co, cs))
                co += cs

        inputLoadSchedule = []
        outputLoadSchedule = []

        # Outer loop over Cout slabs (from cubes), inner over spatial tiles
        for coOff, coSz in co_slices:
            dwTile = HyperRectangle(
                (coOff, 0, 0, 0),
                (coSz, dwShape[1], dwShape[2], dwShape[3]),
            )
            for hoOff, hoSz in h_tiles:
                for woOff, woSz in w_tiles:
                    dyTile = HyperRectangle(
                        (0, coOff, hoOff, woOff),
                        (N_tile, coSz, hoSz, woSz),
                    )
                    xTile, (tpt, tpb, tpl, tpr) = ConvGradWTileConstraintBase.computeInputTileFromGradOutTile(
                        kernel_hw = (dwShape[2], dwShape[3]),
                        pads = pads,
                        strides = strides,
                        inputCSize = Cin_full,
                        gradOutTile = dyTile,
                        inputFull = xFull,
                        gradOutFull = dyFull,
                    )

                    replacements["dim_im_in_x"].append(xTile.dims[2])
                    replacements["dim_im_in_y"].append(xTile.dims[3])
                    replacements["dim_im_out_x"].append(dyTile.dims[2])
                    replacements["dim_im_out_y"].append(dyTile.dims[3])
                    replacements["ch_im_in"].append(Cin_full)
                    replacements["ch_im_out"].append(coSz)
                    replacements["padding_y_top"].append(tpt)
                    replacements["padding_y_bottom"].append(tpb)
                    replacements["padding_x_left"].append(tpl)
                    replacements["padding_x_right"].append(tpr)

                    inputLoadSchedule.append({owner_cls.dataInKey: xTile, owner_cls.gradOutKey: dyTile})
                    outputLoadSchedule.append({owner_cls.weightKey: dwTile})

        return (VariableReplacementScheme(replacements, replacementTypes),
                TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule))


class ConvGradWTileConstraintBase(TileConstraint):
    """
    Base for ConvGradW2D tiling (im2col-style):
      - tile grad_out (dY) over H/W
      - for each dY tile, derive the required input (X) tile (with kernel halo)
      - grad_weight (dW) is NOT tiled (accumulation target is full tensor)
      - unified template padding naming:
          ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
        where:
          x => H dimension  (vertical)  => top/bottom
          y => W dimension  (horizontal)=> left/right

    Tiling regime is dispatched through ``strategies`` (ordered priority list).
    Subclasses override to pick which strategies apply.
    """

    # Default = CinSlice first (perf path for small-spatial / big-channel
    # regular Conv layers), CoutHWSlice as always-feasible fallback.
    # ``PWConvGradWTileConstraint`` and ``DWConvGradW2DTileConstraint`` override
    # this to ``[CoutHWSliceStrategy]`` only — CinSlice's tile schedule
    # assumes the standard dW layout [Cout, Cin/group, P, Q] and breaks for
    # the PW (1x1) and DW ([C, 1, P, Q]) layouts (observed as L1 bank OOB
    # at sim time when CinSlice is dispatched for these subclasses).
    strategies: List = [CinSliceStrategy, CoutHWSliceStrategy]

    # ---- parser/opRep keys (override if needed) ----
    dataInKey = "data_in"  # X (forward input)
    gradOutKey = "grad_out"  # dY
    weightKey = "grad_weight"  # dW (output tensor)

    # ---------------------------
    # 1) Geometrical constraints
    # ---------------------------
    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        tilerModel.addTensorDimToModel(ctxt, xName)
        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dwName)

        group = parseDict.get("group", 1)

        # X, dY are NCHW
        N_x = tilerModel.getTensorDimVar(xName, 0)
        Ci_x = tilerModel.getTensorDimVar(xName, 1)

        N_dy = tilerModel.getTensorDimVar(dyName, 0)
        Co_dy = tilerModel.getTensorDimVar(dyName, 1)

        # dW layout (standard): [C_out, C_in_per_group, P, Q]
        Co_dw = tilerModel.getTensorDimVar(dwName, 0)
        Ci_dw = tilerModel.getTensorDimVar(dwName, 1)

        # batch match
        tilerModel.addConstraint(N_x == N_dy)

        # channel relations
        tilerModel.addConstraint(Co_dy == Co_dw)
        tilerModel.addConstraint(Ci_x == Ci_dw * group)

        return tilerModel

    # -----------------------
    # 2) Policy constraints (dispatched through strategies)
    # -----------------------
    @classmethod
    def _pick_strategy(cls, ctxt: NetworkContext, parseDict: Dict):
        """Pick the best applicable strategy using roofline DMA cost model."""
        applicable = [s for s in cls.strategies if s.applies(cls, ctxt, parseDict)]
        if not applicable:
            return cls.strategies[0] if cls.strategies else None
        if len(applicable) == 1:
            return applicable[0]
        # Use first-applicable (empirically validated priority order).
        # Cost model logged for analysis but does NOT override selection.
        return applicable[0]

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        strat = cls._pick_strategy(ctxt, parseDict)
        return strat.add_constraints(cls, tilerModel, parseDict, ctxt)

    # -----------------------------------
    # 3) Symbolic node representation
    # -----------------------------------
    @classmethod
    def constructSymbolicNodeRep(cls, tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        """
        Template bindings (matches your new template style for ConvGradW):
          - dim_im_out_* / ch_im_out : for grad_out (dY)
          - dim_im_in_*  / ch_im_in  : for input   (X)
          - dim_kernel_* : from dW tensor
          - padding_*    : unified naming
        """
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        symbolic = parseDict.copy()

        # dY tile
        symbolic["dim_im_out_x"] = tilerModel.getTensorDimVar(dyName, 2)  # H_out tile
        symbolic["dim_im_out_y"] = tilerModel.getTensorDimVar(dyName, 3)  # W_out tile
        symbolic["ch_im_out"] = tilerModel.getTensorDimVar(dyName, 1)  # C_out

        # X tile
        symbolic["dim_im_in_x"] = tilerModel.getTensorDimVar(xName, 2)  # H_in tile
        symbolic["dim_im_in_y"] = tilerModel.getTensorDimVar(xName, 3)  # W_in tile
        symbolic["ch_im_in"] = tilerModel.getTensorDimVar(xName, 1)  # C_in

        # Kernel dims from dW: [C_out, C_in_per_group, P, Q]
        symbolic["dim_kernel_x"] = tilerModel.getTensorDimVar(dwName, 2)  # P (H)
        symbolic["dim_kernel_y"] = tilerModel.getTensorDimVar(dwName, 3)  # Q (W)

        return symbolic

    # -------------------------------
    # helpers
    # -------------------------------
    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return -((-a) // b)

    @staticmethod
    def _floor_div(a: int, b: int) -> int:
        return a // b

    @classmethod
    def computeInputTileFromGradOutTile(
            cls,
            kernel_hw: Tuple[int, int],  # (P, Q)
            pads: Tuple[int, int, int, int],  # (t, b, l, r)
            strides: Tuple[int, int],  # (sh, sw)
            inputCSize: int,  # Cin (full)
            gradOutTile: HyperRectangle,  # dY tile (N, Cout, Ho_t, Wo_t)
            inputFull: Tuple[int, int, int, int],  # X full (N, Cin, Hi, Wi)
            gradOutFull: Tuple[int, int, int, int],  # dY full (N, Cout, Ho, Wo)
    ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:
        """
        Given dY tile offsets, compute required X tile:
          h_in in [h_out*sh - pad_top, h_out*sh - pad_top + P)
          w_in in [w_out*sw - pad_left, w_out*sw - pad_left + Q)
        """
        (nOff, _cOff, hoOff, woOff) = gradOutTile.offset
        (nSize, _cSize, hoSize, woSize) = gradOutTile.dims

        pad_top, pad_bottom, pad_left, pad_right = pads
        sh, sw = strides
        P, Q = kernel_hw

        h_in_start = hoOff * sh - pad_top
        w_in_start = woOff * sw - pad_left

        h_in_end = (hoOff + hoSize - 1) * sh - pad_top + P
        w_in_end = (woOff + woSize - 1) * sw - pad_left + Q

        # clamp to X valid range
        h_in_start_c = max(0, h_in_start)
        w_in_start_c = max(0, w_in_start)
        h_in_end_c = min(inputFull[2], h_in_end)
        w_in_end_c = min(inputFull[3], w_in_end)

        hiSize = max(1, h_in_end_c - h_in_start_c)
        wiSize = max(1, w_in_end_c - w_in_start_c)

        xTile = HyperRectangle(
            (nOff, 0, h_in_start_c, w_in_start_c),
            (nSize, inputCSize, hiSize, wiSize),
        )

        # ONNX pads apply only on boundary tiles of dY space
        Hy = gradOutFull[2]
        Wy = gradOutFull[3]

        tile_pad_top = pad_top if hoOff == 0 else 0
        tile_pad_bottom = pad_bottom if (hoOff + hoSize) == Hy else 0
        tile_pad_left = pad_left if woOff == 0 else 0
        tile_pad_right = pad_right if (woOff + woSize) == Wy else 0

        return xTile, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)

    @classmethod
    def extraSerializeChecks(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> None:
        """Hook for DW checks etc."""
        return

    # ---------------------------------------------------
    # 4) serialize: dispatch to the strategy that produced this solution
    # ---------------------------------------------------
    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        # Find which strategy's signature matches the tiler solution; fall back
        # to the first registered strategy if none matches.
        chosen = None
        for strat in cls.strategies:
            if strat.matches_solution(cls, tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt,
                                      operatorRepresentation):
                chosen = strat
                break
        if chosen is None:
            if not cls.strategies:
                raise RuntimeError(f"{cls.__name__}: no tiling strategy configured")
            chosen = cls.strategies[0]

        result = chosen.serialize(cls, tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt,
                                  operatorRepresentation)

        # ── Post-solve roofline cost estimation using actual tile sizes ──
        try:
            import math
            num_tiles = len(absoluteOutputCubes)
            bpe = 4  # FP32
            dyName = operatorRepresentation[cls.gradOutKey]
            xName = operatorRepresentation[cls.dataInKey]
            dwName = operatorRepresentation[cls.weightKey]

            dy_tile = tilingSolution.tensorMemoryConstraints[dyName].memoryConstraints[targetMemLevel].shape
            x_tile = tilingSolution.tensorMemoryConstraints[xName].memoryConstraints[targetMemLevel].shape
            dw_tile = tilingSolution.tensorMemoryConstraints[dwName].memoryConstraints[targetMemLevel].shape

            dy_tile_bytes = math.prod(dy_tile) * bpe
            x_tile_bytes = math.prod(x_tile) * bpe
            dw_tile_bytes = math.prod(dw_tile) * bpe

            dyFull = tuple(ctxt.lookup(dyName).shape)
            dy_hoistable = (tuple(dy_tile) == dyFull)

            tile_tensors = [
                (cls.gradOutKey, dy_tile_bytes, dy_hoistable),
                (cls.dataInKey, x_tile_bytes, False),
                (cls.weightKey, dw_tile_bytes, False),
            ]
            tile_shapes = {
                cls.gradOutKey: tuple(dy_tile),
                cls.dataInKey: tuple(x_tile),
                cls.weightKey: tuple(dw_tile),
            }
            cost = _post_solve_cost(chosen.name, num_tiles, tile_tensors, tile_shapes, ctxt, operatorRepresentation,
                                    cls)
            node = operatorRepresentation.get('nodeName', '?')
            # print(f"[TileCost] {node}: strategy={cost['strategy']} tiles={cost['num_tiles']} "
            #       f"AI={cost['tile_AI']:.2f} attainable={cost['tile_attainable_perf']:.1f}F/cyc "
            #       f"({cost['bound']}-bound) overhead={cost['overhead']:.2f}x "
            #       f"tile_compute={cost['tile_compute_cycles']:.0f} tile_dma={cost['tile_dma_cycles']:.0f} "
            #       f"total={cost['total_cycles']:.0f} ideal={cost['ideal_cycles']:.0f}")
            # for line in cost['breakdown']:
            #     print(line)
        except Exception as e:
            pass  # cost estimation is best-effort

        return result


class ConvGradW2DTileConstraint(ConvGradWTileConstraintBase):
    """Standard ConvGradW2D (non-depthwise).

    Tries CinSlice first (big GEMM K = Hout*Wout, applies when dY fits L1)
    and falls back to CoutHWSlice for layers whose dY exceeds the L1 budget
    (e.g. MobileNetV1 stem: dY = 16x96x96 = 576KB; tiler picks Cout/HW split).
    """
    strategies: List = [CinSliceStrategy, CoutHWSliceStrategy]

    # When CoutHWSlice handles a regular conv, keep Cout full so dW stays a
    # single spatial-reduction slab (memset-once + mm_add accumulation).
    # Forbidding the Cout+HW combination is what fixes the Siracusa CI dW drift;
    # the spatial link still tiles H/W into strips for feasibility.
    coutHWSlice_force_cout_full = True


class PWConvGradWTileConstraint(ConvGradWTileConstraintBase):
    """Pointwise (1x1) ConvGradW — forbid H/W tiling.

    Ideal would be: let the tiler freely pick H/W or C_out (conditional
    template picks memset strategy). That works for simple shapes where the
    tiler commits to ONE axis, but breaks on shapes like MobileNet block_11
    PW (C=128→256, HW=3×3): dW is 128 KB (= full L1), so the tiler is forced
    to mix C_out + HW tiling simultaneously. In that mixed case neither
    memset strategy (per-tile or first-tile-only) is correct without extra
    per-C_out-slice transition tracking. Until the codegen supports that,
    restricting PW to C_out-only keeps the template's per-tile memset
    correct (tiles write disjoint dW slices).

    Strategy: only CoutHWSlice. CinSlice's serialize iterates dW Cin slices,
    which is the wrong axis for PW (1x1 kernel, Cin reduction handled in
    serialize via mm_add across Cin slabs) — sim hits L1 bank OOB.
    """
    strategies: List = [CoutHWSliceStrategy]

    # Small-dW PW convs: let CoutHWSliceStrategy force Cout-full (its dW<=64KB
    # branch). With Cout pinned full there is no Cout tiling, so allowing H/W to
    # tile is the SAFE "HW-only" case (single memset + mm_add accumulation) — not
    # the unimplemented mixed Cout+HW case. This lets the (full-Cin) activation X
    # tile spatially so it fits L1 (e.g. MCUNet conv2d_2 X=[16,48,48]=147KB).
    coutHWSlice_force_cout_full = True

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        super().addPolicyConstraint(tilerModel, parseDict, ctxt)

        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        xBuf = ctxt.lookup(xName)
        dyBuf = ctxt.lookup(dyName)
        dwBuf = ctxt.lookup(dwName)

        dwBytes = 1
        for _d in dwBuf.shape:
            dwBytes *= int(_d)
        dwBytes *= 4

        # Large dW (e.g. MobileNetV1 block_11 PW, dW~128KB): forcing Cout-full would
        # blow L1 and force the broken Cout+HW mixed case, so keep the original
        # full-spatial (Cout-only) restriction. Small dW: Cout-full (from the flag
        # above) + free H/W → spatial tiling, which is what makes MCUNet feasible.
        if dwBytes > 65536:
            tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 2) == dyBuf.shape[2])
            tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 3) == dyBuf.shape[3])
            tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 2) == xBuf.shape[2])
            tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 3) == xBuf.shape[3])

        return tilerModel


class ConvGradBTileConstraint(TileConstraint):
    """
    TileConstraint for ConvGradB: dB[c] = sum_{n,h,w} dY[n,c,h,w]

    Tiles along C (output channels). N, H, W are kept full (reduction dims).
      Input:  grad_out (dY) [N, C, H, W]  — load C-slice per tile
      Output: grad_bias (dB) [C]          — write C-slice per tile
    """

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dyName = parseDict['grad_out']
        dbName = parseDict['grad_bias']

        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dbName)

        dyBuf = ctxt.lookup(dyName)
        N, C, H, W = dyBuf.shape[0], dyBuf.shape[1], dyBuf.shape[2], dyBuf.shape[3]

        # C must match between dY and dB
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == tilerModel.getTensorDimVar(dbName, 0))

        # Keep N, H, W full (reduction dims — cannot split without atomics)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 2) == H)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 3) == W)

        return tilerModel

    @classmethod
    def constructSymbolicNodeRep(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> Dict:
        dyName = parseDict['grad_out']
        dyBuf = ctxt.lookup(dyName)
        N, H, W = dyBuf.shape[0], dyBuf.shape[2], dyBuf.shape[3]

        symbolic = parseDict.copy()
        symbolic['ch_im_out'] = tilerModel.getTensorDimVar(dyName, 1)
        symbolic['batch'] = N
        symbolic['dim_im_out_x'] = H
        symbolic['dim_im_out_y'] = W
        return symbolic

    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        dyName = operatorRepresentation['grad_out']
        dyBuf = ctxt.lookup(dyName)
        N, H, W = dyBuf.shape[0], dyBuf.shape[2], dyBuf.shape[3]

        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, ['grad_out', 'grad_bias'])

        replacements: Dict[str, List[int]] = {'ch_im_out': []}
        replacementTypes = {'ch_im_out': PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for absOut in absoluteOutputCubes:
            dbTile = absOut.rectangle  # 1D: offset=(c_off,), dims=(c_size,)
            c_off = dbTile.offset[0]
            c_size = dbTile.dims[0]

            dyTile = HyperRectangle((0, c_off, 0, 0), (N, c_size, H, W))

            replacements['ch_im_out'].append(c_size)
            inputLoadSchedule.append({'grad_out': dyTile})
            outputLoadSchedule.append({'grad_bias': dbTile})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)
        return variableReplacementSchedule, tilingSchedule


class DWConvGradW2DTileConstraint(ConvGradWTileConstraintBase):
    """
    Depthwise ConvGradW:
      - X:  [N, C, Hi, Wi]
      - dY: [N, C, Ho, Wo]   (Cout == Cin == C)
      - dW: [C, 1, P, Q]

    Strategy: only CoutHWSlice. CinSlice's tile schedule iterates dW
    Cin slices, but DW dW[1] == 1 makes that degenerate; CinSlice's
    serialize also assumes standard [Cout, Cin, P, Q] layout — sim hits
    L1 bank OOB if dispatched here.
    """
    strategies: List = [CoutHWSliceStrategy]

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        tilerModel.addTensorDimToModel(ctxt, xName)
        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dwName)

        # N match
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 0) == tilerModel.getTensorDimVar(dyName, 0))

        # DW dW layout: [C, 1, P, Q]
        C_dw = tilerModel.getTensorDimVar(dwName, 0)
        Cpg_dw = tilerModel.getTensorDimVar(dwName, 1)

        # X and dY channels are both C
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 1) == C_dw)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == C_dw)

        # Cin_per_group must be 1
        tilerModel.addConstraint(Cpg_dw == 1)

        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """DW ConvGradW policy.

        Allows the tiler to pick **C tiling** (preferred when dW is large —
        C slices are disjoint so per-tile memset is trivially correct) OR
        H/W tiling (preferred when dY/X spatial is the L1 bottleneck — the
        conditional template's first-tile-only memset keeps the mm_add
        accumulation across HW tiles correct). DW invariants (Cin==Cout==C,
        dW[1]==1, kernel dims full) are still enforced; C on X/dY is tied
        to C on dW so all three slice together when the tiler picks C
        tiling.
        """
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        xBuf = ctxt.lookup(xName)
        dyBuf = ctxt.lookup(dyName)
        dwBuf = ctxt.lookup(dwName)

        # DW invariants
        # Cin on X == Cout on dY == C on dW (ties channel slicing across
        # all three tensors; the geometrical constraint already enforces
        # this, repeat here as a belt-and-suspenders for the policy solver)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 1) == tilerModel.getTensorDimVar(dwName, 0))
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == tilerModel.getTensorDimVar(dwName, 0))
        # dW[1] must stay 1 (DW weight layout is [C, 1, P, Q])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 1) == 1)
        # Kernel dims full
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 2) == dwBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 3) == dwBuf.shape[3])

        # dY tile spatial dims >= 1
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 2) >= 1)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 3) >= 1)

        return tilerModel

    @classmethod
    def extraSerializeChecks(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> None:
        xName = operatorRepresentation[cls.dataInKey]
        dyName = operatorRepresentation[cls.gradOutKey]
        dwName = operatorRepresentation[cls.weightKey]

        xFull = tuple(ctxt.lookup(xName).shape)
        dyFull = tuple(ctxt.lookup(dyName).shape)
        dwShape = tuple(ctxt.lookup(dwName).shape)

        Cin = xFull[1]
        Cout = dyFull[1]
        assert Cin == Cout, f"DWConvGradW expects Cin==Cout, got Cin={Cin}, Cout={Cout}"
        assert dwShape[0] == Cin, f"DWConvGradW expects dW[0]==C, got dW[0]={dwShape[0]} vs C={Cin}"
        assert dwShape[1] == 1, f"DWConvGradW expects dW[1]==1, got dW[1]={dwShape[1]}"
        return
