/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel/PULPPWConv1x1.h"

#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nn_utils.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif

#define PW_MIN(a, b) (((a) < (b)) ? (a) : (b))

static inline uint8_t __attribute__((always_inline))
DeeployPULP_pw_requant_u8(int32_t acc, int32_t kappa, int32_t lambda,
                          uint16_t shift) {
  return (uint8_t)clip8(((kappa * acc) + lambda) >> shift);
}

/*
 * As above, but blocked four pixels by two output channels instead of two by
 * four, so each block's four results per channel land on consecutive bytes of a
 * channels-first output. The load-to-multiply ratio is unchanged -- four input
 * and two weight vectors feed eight accumulators either way -- and the
 * channels-first result is what the next depthwise kernel reads, which removes
 * the transpose that otherwise sits between them.
 */
void DeeployPULP_PW_Conv2d_1x1_CHWOut_u8_u8_i8(
    uint8_t *pIn, uint8_t *pIm2ColBuffer, int8_t *pBias, uint8_t *pOut,
    int8_t *pWeight, int32_t *pKappa, int32_t *pLambda, uint16_t out_mult,
    uint16_t out_shift, uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in,
    uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
    uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t padding_y_top,
    uint16_t padding_y_bottom, uint16_t padding_x_left,
    uint16_t padding_x_right, uint16_t stride_x, uint16_t stride_y,
    uint8_t flag_relu, uint8_t flag_batch_norm) {

  const int nPix = dim_out_x * dim_out_y;
  const int nCol = ch_in >> 2;
  const int core_id = pi_core_id();

  const int nPair = ch_out >> 1;
  const int chunk = (nPair + NUM_CORES - 1) / NUM_CORES;
  const int g0 = PW_MIN(chunk * core_id, nPair) << 1;
  const int g1 = PW_MIN(chunk * (core_id + 1), nPair) << 1;

  for (int g = g0; g < g1; g += 2) {
    const int8_t *const w0 = pWeight + g * ch_in;
    const int8_t *const w1 = w0 + ch_in;
    const int32_t k0 = pKappa[g], k1 = pKappa[g + 1];
    const int32_t l0 = pLambda[g], l1 = pLambda[g + 1];
    uint8_t *const o0 = pOut + g * nPix;
    uint8_t *const o1 = o0 + nPix;

    int p = 0;
    for (; p + 4 <= nPix; p += 4) {
      const int8_t *a0 = w0, *a1 = w1;
      const uint8_t *b0 = pIn + p * ch_in;
      const uint8_t *b1 = b0 + ch_in, *b2 = b1 + ch_in, *b3 = b2 + ch_in;
      int s00 = 0, s01 = 0, s02 = 0, s03 = 0;
      int s10 = 0, s11 = 0, s12 = 0, s13 = 0;

      for (int j = 0; j < nCol; j++) {
        const v4s vA0 = *((v4s *)a0);
        const v4s vA1 = *((v4s *)a1);
        const v4u vB0 = *((v4u *)b0);
        const v4u vB1 = *((v4u *)b1);
        const v4u vB2 = *((v4u *)b2);
        const v4u vB3 = *((v4u *)b3);

        s00 = SumDotp4(vB0, vA0, s00);
        s01 = SumDotp4(vB1, vA0, s01);
        s02 = SumDotp4(vB2, vA0, s02);
        s03 = SumDotp4(vB3, vA0, s03);
        s10 = SumDotp4(vB0, vA1, s10);
        s11 = SumDotp4(vB1, vA1, s11);
        s12 = SumDotp4(vB2, vA1, s12);
        s13 = SumDotp4(vB3, vA1, s13);

        a0 += 4;
        a1 += 4;
        b0 += 4;
        b1 += 4;
        b2 += 4;
        b3 += 4;
      }

      o0[p] = DeeployPULP_pw_requant_u8(s00, k0, l0, out_shift);
      o0[p + 1] = DeeployPULP_pw_requant_u8(s01, k0, l0, out_shift);
      o0[p + 2] = DeeployPULP_pw_requant_u8(s02, k0, l0, out_shift);
      o0[p + 3] = DeeployPULP_pw_requant_u8(s03, k0, l0, out_shift);
      o1[p] = DeeployPULP_pw_requant_u8(s10, k1, l1, out_shift);
      o1[p + 1] = DeeployPULP_pw_requant_u8(s11, k1, l1, out_shift);
      o1[p + 2] = DeeployPULP_pw_requant_u8(s12, k1, l1, out_shift);
      o1[p + 3] = DeeployPULP_pw_requant_u8(s13, k1, l1, out_shift);
    }

    for (; p < nPix; p++) {
      const int8_t *a0 = w0, *a1 = w1;
      const uint8_t *b0 = pIn + p * ch_in;
      int s00 = 0, s10 = 0;
      for (int j = 0; j < nCol; j++) {
        const v4u vB0 = *((v4u *)b0);
        s00 = SumDotp4(vB0, *((v4s *)a0), s00);
        s10 = SumDotp4(vB0, *((v4s *)a1), s10);
        a0 += 4;
        a1 += 4;
        b0 += 4;
      }
      o0[p] = DeeployPULP_pw_requant_u8(s00, k0, l0, out_shift);
      o1[p] = DeeployPULP_pw_requant_u8(s10, k1, l1, out_shift);
    }
  }

  /* an odd channel count leaves one channel over; it is cheaper to give it to a
     single core than to renumber the split */
  if ((ch_out & 1) && core_id == 0) {
    const int g = ch_out - 1;
    const int8_t *const w0 = pWeight + g * ch_in;
    uint8_t *const o0 = pOut + g * nPix;
    for (int p = 0; p < nPix; p++) {
      const int8_t *a0 = w0;
      const uint8_t *b0 = pIn + p * ch_in;
      int s00 = 0;
      for (int j = 0; j < nCol; j++) {
        s00 = SumDotp4(*((v4u *)b0), *((v4s *)a0), s00);
        a0 += 4;
        b0 += 4;
      }
      o0[p] = DeeployPULP_pw_requant_u8(s00, pKappa[g], pLambda[g], out_shift);
    }
  }

  pi_cl_team_barrier();
}
