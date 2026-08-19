/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_PULP_STEM_CONV_3X3_HEADER_
#define __DEEPLOY_BASIC_MATH_PULP_STEM_CONV_3X3_HEADER_

#include <stdint.h>

void DeeployPULP_Conv2d_3x3_CHW_u8_u8_i8(
    uint8_t *pIn, uint8_t *pIm2ColBuffer, int8_t *pBias, uint8_t *pOut,
    int8_t *pWeight, int32_t *pKappa, int32_t *pLambda, uint16_t out_mult,
    uint16_t out_shift, uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in,
    uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
    uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t padding_y_top,
    uint16_t padding_y_bottom, uint16_t padding_x_left,
    uint16_t padding_x_right, uint16_t stride_x, uint16_t stride_y,
    uint8_t flag_relu, uint8_t flag_batch_norm);

#endif
