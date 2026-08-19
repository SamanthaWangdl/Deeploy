/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel/PULPStemConv3x3.h"

#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nn_utils.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif

#define STEM_MIN(a, b) (((a) < (b)) ? (a) : (b))

static inline uint8_t __attribute__((always_inline))
DeeployPULP_stem_requant_u8(int32_t acc, int32_t kappa, int32_t lambda,
                            uint16_t shift) {
  return (uint8_t)clip8(((kappa * acc) + lambda) >> shift);
}

/*
 * Convolution over a channels-first input producing a channels-first output,
 * for the three-channel stem of an image network. Both layouts are what the
 * neighbours already use -- the input arrives NCHW and the depthwise kernel
 * that follows reads NCHW -- so the two transposes around this layer disappear.
 *
 * pulp_nn_conv instead gathers a 3x3x3 im2col column per output pixel and runs
 * a matmul whose reduction length, 27, is not a multiple of four. Here the
 * three taps along a row are contiguous within one channel plane, so the window
 * is nine v4s dot products with the fourth lane zeroed, and no copy.
 *
 * Anything but three input channels with a padded 3x3 kernel falls to the
 * reference loop, which is correct for any shape.
 */
/*
 * One input channel's contribution to a column of outputs. mode 0 seeds the
 * column, 1 adds to it, 2 adds and scales straight into the output, so neither
 * a zeroing pass nor a separate scaling pass over the column is needed.
 * Inlined, so the mode folds away and each call site is one of the three loops.
 */
static inline void __attribute__((always_inline))
DeeployPULP_stem_pass(const uint8_t *pr, int32_t *colAcc, uint8_t *pw, v4s C0,
                      v4s C1, v4s C2, int W, int H, int OW, int OH,
                      int stride_y, int padY, int32_t kk, int32_t ll,
                      uint16_t shift, const int mode) {

  const uint8_t *const rowBase = pr;
  v4u V0 = (v4u){0, 0, 0, 0};
  v4u V1 = *(v4u *)pr;
  pr += W;
  v4u V2 = *(v4u *)pr;
  pr += W;

  /* rows stride_y*y+2 stay inside up to here, so the loop needs no test */
  const int yFull = STEM_MIN((H - 1) / stride_y, OH);
  int y = 0;
  for (; y < yFull; y++) {
    int a = SumDotp4(V0, C0, 0);
    a = SumDotp4(V1, C1, a);
    a = SumDotp4(V2, C2, a);
    if (mode == 0) {
      colAcc[y] = a;
    } else if (mode == 1) {
      colAcc[y] += a;
    } else {
      *pw = DeeployPULP_stem_requant_u8(colAcc[y] + a, kk, ll, shift);
      pw += OW;
    }
    V0 = V2;
    V1 = *(v4u *)pr;
    pr += W;
    V2 = *(v4u *)pr;
    pr += W;
  }
  for (; y < OH; y++) {
    const int ry = y * stride_y - padY;
    int a = SumDotp4(V0, C0, 0);
    if (ry + 1 < H) {
      a = SumDotp4(*(v4u *)(rowBase + (size_t)(ry + 1) * W), C1, a);
    }
    if (ry + 2 < H) {
      a = SumDotp4(*(v4u *)(rowBase + (size_t)(ry + 2) * W), C2, a);
    }
    if (mode == 0) {
      colAcc[y] = a;
    } else if (mode == 1) {
      colAcc[y] += a;
    } else {
      *pw = DeeployPULP_stem_requant_u8(colAcc[y] + a, kk, ll, shift);
      pw += OW;
    }
    V0 = (v4u){0, 0, 0, 0};
  }
}

void DeeployPULP_Conv2d_3x3_CHW_u8_u8_i8(
    uint8_t *pIn, uint8_t *pIm2ColBuffer, int8_t *pBias, uint8_t *pOut,
    int8_t *pWeight, int32_t *pKappa, int32_t *pLambda, uint16_t out_mult,
    uint16_t out_shift, uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in,
    uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
    uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t padding_y_top,
    uint16_t padding_y_bottom, uint16_t padding_x_left,
    uint16_t padding_x_right, uint16_t stride_x, uint16_t stride_y,
    uint8_t flag_relu, uint8_t flag_batch_norm) {

  const int W = dim_in_x, H = dim_in_y;
  const int OW = dim_out_x, OH = dim_out_y;
  const int plane = W * H, nPix = OW * OH;
  const int kW = dim_kernel_x, kH = dim_kernel_y;
  const int padX = padding_x_left, padY = padding_y_top;

  const int fast =
      (ch_in == 3 && kW == 3 && kH == 3 && padX == 1 && padY == 1 && W >= 4);

  const int core_id = pi_core_id();
  const int chunk = (ch_out + NUM_CORES - 1) / NUM_CORES;
  const int co0 = STEM_MIN(chunk * core_id, ch_out);
  const int co1 = STEM_MIN(chunk * (core_id + 1), ch_out);

  for (int co = co0; co < co1; co++) {
    /* One column of 32-bit accumulators per core, with the input channels as
       the outer loop, after the Autotiler's KerParConv3x3Stride2_SQ8. Summing
       the channels in registers instead needs nine taps and nine rotating rows
       live at once, which no register file here has. The cost is a pass per
       channel, so the pass below carries nothing it does not need: the row
       bounds are hoisted out of it, the rows walk by pointer, and each output
       costs two loads because consecutive stride-2 windows share a row. */
    int32_t *const colAcc = (int32_t *)pIm2ColBuffer + (size_t)core_id * OH;

    for (int x = 0; x < OW; x++) {
      const int base = x * stride_x - padX;
      const int interior = fast && base >= 0 && base <= W - 4;

      for (int y = 0; y < OH; y++) {
        colAcc[y] = 0;
      }

      const int32_t kk = pKappa[co], ll = pLambda[co];
      uint8_t *const pw = pOut + (size_t)co * nPix + x;

      if (interior) {
        for (int ci = 0; ci < ch_in; ci++) {
          const int8_t *const w = pWeight + (size_t)(co * ch_in + ci) * kW * kH;
          const uint8_t *const pr = pIn + (size_t)ci * plane + base;
          const v4s C0 = (v4s){w[0], w[1], w[2], 0};
          const v4s C1 = (v4s){w[3], w[4], w[5], 0};
          const v4s C2 = (v4s){w[6], w[7], w[8], 0};

          if (ci == 0) {
            DeeployPULP_stem_pass(pr, colAcc, pw, C0, C1, C2, W, H, OW, OH,
                                  stride_y, padY, kk, ll, out_shift, 0);
          } else if (ci < ch_in - 1) {
            DeeployPULP_stem_pass(pr, colAcc, pw, C0, C1, C2, W, H, OW, OH,
                                  stride_y, padY, kk, ll, out_shift, 1);
          } else {
            DeeployPULP_stem_pass(pr, colAcc, pw, C0, C1, C2, W, H, OW, OH,
                                  stride_y, padY, kk, ll, out_shift, 2);
          }
        }
        continue;
      }

      for (int y = 0; y < OH; y++) {
        colAcc[y] = 0;
      }
      for (int ci = 0; ci < ch_in; ci++) {
        const int8_t *const w = pWeight + (size_t)(co * ch_in + ci) * kW * kH;
        const uint8_t *const inp = pIn + (size_t)ci * plane;
        for (int y = 0; y < OH; y++) {
          const int ry = y * stride_y - padY;
          int a = 0;
          for (int ky = 0; ky < kH; ky++) {
            const int iy = ry + ky;
            if (iy < 0 || iy >= H) {
              continue;
            }
            for (int kx = 0; kx < kW; kx++) {
              const int ix = base + kx;
              if (ix < 0 || ix >= W) {
                continue;
              }
              a += (int)inp[(size_t)iy * W + ix] * (int)w[ky * kW + kx];
            }
          }
          colAcc[y] += a;
        }
      }
      {
        uint8_t *p = pw;
        for (int y = 0; y < OH; y++) {
          *p = DeeployPULP_stem_requant_u8(colAcc[y], kk, ll, out_shift);
          p += OW;
        }
      }
    }
  }

  pi_cl_team_barrier();
}
