/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel/PULPDWConv3x3.h"

#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nn_utils.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif

#define DW3X3_MIN(a, b) (((a) < (b)) ? (a) : (b))

/*
 * Shape-specialised 3x3 / stride 1 / pad 1 depthwise convolution, following the
 * structure of the GAP9 SDK Autotiler's KerConvDW3x3Stride1_Body_SQ8
 * (tools/autotiler_v3/CNN_Libraries_SQ8, Apache-2.0): the 9 taps sit in three
 * v4s registers with lane 3 zeroed, the three input rows rotate through
 * registers so each output pixel costs one load, and the borders are folded
 * into shifted weight vectors. Any other shape tail-calls
 * pulp_nn_depthwise_u8_u8_i8.
 */

/* rows narrower than one v4u are assembled byte-wise, so the load never runs
   past the row */
static inline v4u __attribute__((always_inline))
DeeployPULP_dw_load3(const uint8_t *p) {
  v4u v = (v4u){0, 0, 0, 0};
  v[0] = p[0];
  v[1] = p[1];
  v[2] = p[2];
  return v;
}

static inline uint8_t __attribute__((always_inline))
DeeployPULP_dw_requant_u8(int32_t acc, int32_t kappa, int32_t lambda,
                          uint16_t shift) {
  return (uint8_t)clip8(((kappa * acc) + lambda) >> shift);
}

void DeeployPULP_DW_Conv2d_3x3_u8_u8_i8(
    uint8_t *pIn, uint8_t *pIm2ColBuffer, int8_t *pBias, uint8_t *pOut,
    int8_t *pWeight, int8_t *pWtBuffer, int32_t *pKappa, int32_t *pLambda,
    uint16_t out_mult, uint16_t out_shift, uint16_t dim_in_x, uint16_t dim_in_y,
    uint16_t ch_in, uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
    uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t padding_y_top,
    uint16_t padding_y_bottom, uint16_t padding_x_left,
    uint16_t padding_x_right, uint16_t stride_x, uint16_t stride_y,
    uint8_t flag_relu, uint8_t flag_batch_norm) {

  if (!(dim_kernel_x == 3 && dim_kernel_y == 3 && stride_x == stride_y &&
        (stride_x == 1 || stride_x == 2) && padding_x_left == 1 &&
        padding_x_right == 1 && padding_y_top == 1 && padding_y_bottom == 1 &&
        dim_out_x == (dim_in_x + stride_x - 1) / stride_x &&
        dim_out_y == (dim_in_y + stride_y - 1) / stride_y && dim_in_x >= 3 &&
        dim_in_y >= 2 && pBias == NULL && flag_relu && flag_batch_norm)) {
    pulp_nn_depthwise_u8_u8_i8(
        pIn, pIm2ColBuffer, pBias, pOut, pWeight, pWtBuffer, pKappa, pLambda,
        out_mult, out_shift, dim_in_x, dim_in_y, ch_in, dim_out_x, dim_out_y,
        ch_out, dim_kernel_x, dim_kernel_y, padding_y_top, padding_y_bottom,
        padding_x_left, padding_x_right, stride_x, stride_y, flag_relu,
        flag_batch_norm);
    return;
  }

  const uint8_t core_id = pi_core_id();
  const int chunk =
      (ch_out >> __builtin_ctz(NUM_CORES)) + ((ch_out & (NUM_CORES - 1)) != 0);
  const int start_channel = DW3X3_MIN(chunk * core_id, ch_out);
  const int stop_channel = DW3X3_MIN(start_channel + chunk, ch_out);

  const v4u ZERO = (v4u){0, 0, 0, 0};
  const int W = dim_in_x, H = dim_in_y, OS = dim_out_x * ch_out;
  const int plane = dim_in_x * dim_in_y;

  for (int c = start_channel; c < stop_channel; c++) {
    const uint8_t *inp = pIn + c * plane;
    const int8_t *wt = pWeight + c * 9;
    const int32_t kk = pKappa[c], ll = pLambda[c];

    /* only the interior column's taps stay live across the x loop: twelve
       vectors exhaust the register file and spill w0..w2, kappa, lambda and the
       shift back into every y iteration. The three border columns rebuild
       theirs below, where it costs nothing. */
    const v4s C0 = (v4s){wt[0], wt[1], wt[2], 0};
    const v4s C1 = (v4s){wt[3], wt[4], wt[5], 0};
    const v4s C2 = (v4s){wt[6], wt[7], wt[8], 0};

    if (W == 3) {
      /* every column reads at offset 0; the third one drops tap 2 */
      /* every weight vector has lane 3 zeroed, so a word load may read one byte
         past the row without changing the result -- only the last row of the
         last channel could run past the tile itself */
      const int wordSafe = (inp + 2 * W + 4) <= (pIn + (size_t)ch_out * plane);
      for (int x = 0; x < dim_out_x; x++) {
        v4s w0, w1, w2;
        if (x * stride_x == 1) {
          w0 = C0;
          w1 = C1;
          w2 = C2;
        } else if (x * stride_x == 0) {
          w0 = (v4s){wt[1], wt[2], 0, 0};
          w1 = (v4s){wt[4], wt[5], 0, 0};
          w2 = (v4s){wt[7], wt[8], 0, 0};
        } else {
          w0 = (v4s){0, wt[0], wt[1], 0};
          w1 = (v4s){0, wt[3], wt[4], 0};
          w2 = (v4s){0, wt[6], wt[7], 0};
        }
        uint8_t *po = pOut + c + x * ch_out;
        for (int y = 0; y < dim_out_y; y++) {
          const int r = y * stride_y - 1;
          v4u V0, V1, V2;
          if (wordSafe) {
            V0 = (r >= 0) ? *(v4u *)(inp + r * W) : ZERO;
            V1 = (r + 1 < H) ? *(v4u *)(inp + (r + 1) * W) : ZERO;
            V2 = (r + 2 < H) ? *(v4u *)(inp + (r + 2) * W) : ZERO;
          } else {
            V0 = (r >= 0) ? DeeployPULP_dw_load3(inp + r * W) : ZERO;
            V1 = (r + 1 < H) ? DeeployPULP_dw_load3(inp + (r + 1) * W) : ZERO;
            V2 = (r + 2 < H) ? DeeployPULP_dw_load3(inp + (r + 2) * W) : ZERO;
          }
          int acc = SumDotp4(V0, w0, 0);
          acc = SumDotp4(V1, w1, acc);
          acc = SumDotp4(V2, w2, acc);
          *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
          po += OS;
        }
      }
      continue;
    }

    for (int x = 0; x < dim_out_x; x++) {
      /* first tap of this output column; the four weight sets below cover
         base < 0, an interior read, and the two reads clamped to W-4 */
      const int base = x * stride_x - 1;
      v4s w0, w1, w2;
      int off;
      if (base >= 0 && base <= W - 4) {
        w0 = C0;
        w1 = C1;
        w2 = C2;
        off = base;
      } else if (base < 0) {
        /* x == 0: read 4B at 0, tap 0 sits on the pad -> drop it */
        w0 = (v4s){wt[1], wt[2], 0, 0};
        w1 = (v4s){wt[4], wt[5], 0, 0};
        w2 = (v4s){wt[7], wt[8], 0, 0};
        off = 0;
      } else if (base == W - 3) {
        /* x == W-2: read 4B at W-4 so the load never runs past the row */
        w0 = (v4s){0, wt[0], wt[1], wt[2]};
        w1 = (v4s){0, wt[3], wt[4], wt[5]};
        w2 = (v4s){0, wt[6], wt[7], wt[8]};
        off = W - 4;
      } else {
        /* x == W-1: read 4B at W-4, tap 2 sits on the pad -> drop it */
        w0 = (v4s){0, 0, wt[0], wt[1]};
        w1 = (v4s){0, 0, wt[3], wt[4]};
        w2 = (v4s){0, 0, wt[6], wt[7]};
        off = W - 4;
      }

      const uint8_t *pi = inp + off;
      uint8_t *po = pOut + c + x * ch_out;

      if (stride_y == 2) {
        /* window for output row y is input rows 2y-1, 2y, 2y+1, so consecutive
           windows overlap by one row: V2 becomes the next V0. Rows 2y+1 stay in
           range up to H/2, so the main loop below needs no bounds test */
        const uint8_t *pr = pi;
        const int yFull = DW3X3_MIN(H >> 1, dim_out_y);
        v4u V0 = ZERO;
        v4u V1 = *(v4u *)pr;
        v4u V2 = *(v4u *)(pr + W);
        int y = 0;
        /* unrolled by two so the window's bottom row becomes the next window's
           top by renaming rather than a register move */
        /* one iteration short of the end: this body reads the next window's
           rows before it knows there is one, and at the end those would be
           outside the plane. The row peeled off below closes the gap without
           reading ahead. */
        for (; y + 3 <= yFull; y += 2) {
          int acc = SumDotp4(V0, w0, 0);
          acc = SumDotp4(V1, w1, acc);
          acc = SumDotp4(V2, w2, acc);
          *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
          po += OS;
          pr += 2 * W;
          const v4u U1 = *(v4u *)pr;
          const v4u U2 = *(v4u *)(pr + W);
          acc = SumDotp4(V2, w0, 0);
          acc = SumDotp4(U1, w1, acc);
          acc = SumDotp4(U2, w2, acc);
          *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
          po += OS;
          pr += 2 * W;
          V0 = U2;
          V1 = *(v4u *)pr;
          V2 = *(v4u *)(pr + W);
        }
        for (; y + 1 < yFull; y++) {
          int acc = SumDotp4(V0, w0, 0);
          acc = SumDotp4(V1, w1, acc);
          acc = SumDotp4(V2, w2, acc);
          *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
          po += OS;
          pr += 2 * W;
          V0 = V2;
          V1 = *(v4u *)pr;
          V2 = *(v4u *)(pr + W);
        }
        if (y < yFull) {
          int acc = SumDotp4(V0, w0, 0);
          acc = SumDotp4(V1, w1, acc);
          acc = SumDotp4(V2, w2, acc);
          *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
          po += OS;
          V0 = V2;
          y++;
        }
        for (; y < dim_out_y; y++) {
          /* an odd input height leaves the bottom row of the last window on the
             pad; V1/V2 were speculatively loaded above, so rebuild them here */
          const int r1 = 2 * y, r2 = 2 * y + 1;
          V1 = (r1 < H) ? *(v4u *)(pi + (size_t)r1 * W) : ZERO;
          V2 = (r2 < H) ? *(v4u *)(pi + (size_t)r2 * W) : ZERO;
          int acc = SumDotp4(V0, w0, 0);
          acc = SumDotp4(V1, w1, acc);
          acc = SumDotp4(V2, w2, acc);
          *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
          po += OS;
          V0 = V2;
        }
        continue;
      }

      v4u V0 = ZERO; /* input row -1 is the top pad          */
      v4u V1 = *(v4u *)pi;
      pi += W; /* input row 0                          */

      /* unrolled by three so the three-row window rotates by renaming instead
         of the register moves the rolling form compiles to */
      int y = 0;
      for (; y + 3 <= H - 1; y += 3) {
        const v4u A = *(v4u *)pi;
        const v4u B = *(v4u *)(pi + W);
        const v4u C = *(v4u *)(pi + 2 * W);
        pi += 3 * W;
        int acc = SumDotp4(V0, w0, 0);
        acc = SumDotp4(V1, w1, acc);
        acc = SumDotp4(A, w2, acc);
        *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
        po += OS;
        acc = SumDotp4(V1, w0, 0);
        acc = SumDotp4(A, w1, acc);
        acc = SumDotp4(B, w2, acc);
        *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
        po += OS;
        acc = SumDotp4(A, w0, 0);
        acc = SumDotp4(B, w1, acc);
        acc = SumDotp4(C, w2, acc);
        *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
        po += OS;
        V0 = B;
        V1 = C;
      }
      for (; y < H - 1; y++) {
        const v4u V2 = *(v4u *)pi;
        pi += W; /* the only load per output pixel     */
        int acc = SumDotp4(V0, w0, 0);
        acc = SumDotp4(V1, w1, acc);
        acc = SumDotp4(V2, w2, acc);
        *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
        po += OS;
        V0 = V1;
        V1 = V2; /* two of the three rows are reused next pixel    */
      }
      { /* last output row: input row H is the bottom pad, w2 contributes 0 */
        int acc = SumDotp4(V0, w0, 0);
        acc = SumDotp4(V1, w1, acc);
        *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
      }
    }
  }
  pi_cl_team_barrier();
}
