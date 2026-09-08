/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_CONVGRAD_KERNEL_HEADER_
#define __DEEPLOY_MATH_CONVGRAD_KERNEL_HEADER_

#include "DeeployPULPMath.h"
#include "pmsis.h"

// ============================================================================
// pulp-trainlib interface structs (forward declarations to avoid pulling in
// pulp_train_defines.h which conflicts with Deeploy headers)
// ============================================================================

struct blob {
  float *data;
  float *diff;
  int dim;
  int W;
  int H;
  int C;
};

struct Conv2D_args {
  struct blob *input;
  struct blob *coeff;
  struct blob *bias;
  struct blob *output;
  int Lpad;
  int Rpad;
  int Upad;
  int Dpad;
  int stride_h;
  int stride_w;
  float *i2c_buffer;
  float *bt_buffer;
  int skip_wg_grad;
  int skip_in_grad;
  int HWC;
  int opt_matmul_type_fw;
  int opt_matmul_type_wg;
  int opt_matmul_type_ig;
  int USE_BIASES;
  int USE_IM2COL;
  int USE_DMA_IM2COL;
  int offset_in_h;
  int offset_in_w;
  int offset_out_h;
  int offset_out_w;
};

struct DepthWise_Conv_args {
  struct blob *input;
  struct blob *coeff;
  struct blob *output;
  int stride_h;
  int stride_w;
  int Lpad;
  int Rpad;
  int Upad;
  int Dpad;
  int skip_wg_grad;
  int skip_in_grad;
  int HWC;
  int offset_in_h;
  int offset_in_w;
  int offset_out_h;
  int offset_out_w;
};

struct PointWise_Conv_args {
  struct blob *input;
  struct blob *coeff;
  struct blob *output;
  float *transpose_buffer;
  int skip_wg_grad;
  int skip_in_grad;
  int HWC;
  int opt_matmul_type_fw;
  int opt_matmul_type_wg;
  int opt_matmul_type_ig;
};

// pulp-trainlib dispatch functions
void pulp_conv2d_fp32_fw_cl(void *Conv2D_args);
void pulp_conv_dw_fp32_fw_cl(void *DepthWise_Conv_args);
void pulp_conv2d_fp32_bw_param_grads_cl(void *Conv2D_args);
void pulp_conv2d_fp32_bw_input_grads_cl(void *Conv2D_args);
void pulp_conv2d_fp32_bw_input_grads_tiled_cl(void *Conv2D_args);
void pulp_conv_dw_fp32_bw_param_grads_cl(void *DepthWise_Conv_args);
void pulp_conv_dw_fp32_bw_input_grads_tiled_cl(void *DepthWise_Conv_args);
void pulp_conv_pw_fp32_bw_param_grads_cl(void *PointWise_Conv_args);

// ============================================================================
// Utility
// ============================================================================

static inline int32_t max_i32(int32_t a, int32_t b) { return (a > b) ? a : b; }
static inline int32_t min_i32(int32_t a, int32_t b) { return (a < b) ? a : b; }

// ============================================================================
// Regular (Dense) Conv — GradW
// ============================================================================

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW_Im2Col(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float *__restrict__ ctxtBuffer, uint32_t ctxtBufferSize);

// ============================================================================
// Regular (Dense) Conv — GradX
// ============================================================================

// Tiled scatter-add ConvGradX with tile offsets (ForkTransformer)
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_scatter_tiled(
    const float *__restrict__ pGradOut, uint32_t dim_im_out_x,
    uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in, uint32_t dim_kernel_x,
    uint32_t dim_kernel_y, uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn, uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_x_left, uint32_t padding_x_right, uint32_t padding_y_top,
    uint32_t padding_y_bottom, uint16_t offset_grad_in_h,
    uint16_t offset_grad_in_w, uint16_t offset_grad_out_h,
    uint16_t offset_grad_out_w);

// Tiled im2col+GEMM with co_block (ForkTransformer)
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_Im2Col_tiled(
    const float *__restrict__ pGradOut, uint32_t dim_im_out_x,
    uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in, uint32_t dim_kernel_x,
    uint32_t dim_kernel_y, uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn, uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_y_top, uint32_t padding_y_bottom, uint32_t padding_x_left,
    uint32_t padding_x_right, uint16_t offset_grad_in_h,
    uint16_t offset_grad_in_w, uint16_t offset_grad_out_h,
    uint16_t offset_grad_out_w, float *__restrict__ ctxtBuffer,
    uint32_t ctxtBufferSize, float *__restrict__ btBuffer,
    uint32_t btBufferSize);

// ============================================================================
// Depthwise Conv — GradW (ClusterTransformer)
// ============================================================================

void PULP_DWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

// ============================================================================
// Depthwise Conv — GradX (trainlib gather kernel, ClusterTransformer)
// ============================================================================

void PULP_DWConvGradX2d_fp32_fp32_fp32_CHW_trainlib_tiled(
    const float *__restrict__ pGradOut, uint32_t dim_im_out_x,
    uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in, uint32_t dim_kernel_x,
    uint32_t dim_kernel_y, uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn, uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_x_left, uint32_t padding_x_right, uint32_t padding_y_top,
    uint32_t padding_y_bottom, uint16_t offset_grad_in_h,
    uint16_t offset_grad_in_w, uint16_t offset_grad_out_h,
    uint16_t offset_grad_out_w);

// ============================================================================
// Pointwise Conv — GradW (ClusterTransformer)
// ============================================================================

void PULP_PWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, float *__restrict__ pGradWeight);

// ============================================================================
// Pointwise Conv — GradX (direct AXPY, ClusterTransformer)
// ============================================================================

void PULP_PWConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in);

#endif // __DEEPLOY_MATH_CONVGRAD_KERNEL_HEADER_
