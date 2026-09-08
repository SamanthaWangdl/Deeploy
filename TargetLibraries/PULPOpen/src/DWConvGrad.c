/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

// ============================================================================
// Depthwise Convolution Gradient Kernels
// ============================================================================

void PULP_DWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
  // Supports depthwise convolution with multiplier
  // For depthwise: groups = C_in, C_out = C_in * multiplier
  // Weight shape: [C_out, 1, P, Q]

  uint32_t gradw_elems = C_out * (C_in / C_out) * P * Q;

  struct blob input_blob = {0};
  struct blob coeff_blob = {0};
  struct blob output_blob = {0};

  input_blob.data = (float *)pInput;
  input_blob.diff = NULL;
  input_blob.W = (int)W_in;
  input_blob.H = (int)H_in;
  input_blob.C = (int)C_in;
  input_blob.dim = (int)(C_in * H_in * W_in);

  coeff_blob.data = NULL;
  coeff_blob.diff = (float *)pGradWeight;
  coeff_blob.W = (int)Q;
  coeff_blob.H = (int)P;
  coeff_blob.C = (int)C_out; // Fixed: should be C_out for DW with multiplier
  coeff_blob.dim = (int)(C_out * P * Q); // Fixed: total weight elements

  output_blob.data = NULL;
  output_blob.diff = (float *)pGradOut;
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)C_out;
  output_blob.dim = (int)(C_out * H_out * W_out);

  struct DepthWise_Conv_args dw_args;
  memset(&dw_args, 0, sizeof(dw_args));

  dw_args.input = &input_blob;
  dw_args.coeff = &coeff_blob;
  dw_args.output = &output_blob;

  dw_args.stride_h = (int)SP;
  dw_args.stride_w = (int)SQ;

  dw_args.Lpad = (int)pad_left;
  dw_args.Rpad = (int)pad_right;
  dw_args.Upad = (int)pad_top;
  dw_args.Dpad = (int)pad_bottom;

  dw_args.skip_wg_grad = 0;
  dw_args.skip_in_grad = 1;
  dw_args.HWC = 0;
  pulp_conv_dw_fp32_bw_param_grads_cl(&dw_args);
}

void PULP_DWConvGradX2d_fp32_fp32_fp32_CHW_trainlib_tiled(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, // H_out (tile)
    uint32_t dim_im_out_y, // W_out (tile)
    uint32_t ch_im_out,    // C_out (full)
    const float *__restrict__ pWeight,
    uint32_t ch_im_in,     // C_in (full)
    uint32_t dim_kernel_x, // P (kernel H)
    uint32_t dim_kernel_y, // Q (kernel W)
    uint32_t stride_h, uint32_t stride_w, float *__restrict__ pGradIn,
    uint32_t dim_im_in_x,      // H_in (tile)
    uint32_t dim_im_in_y,      // W_in (tile)
    uint32_t padding_x_left,   // pad_top  (ONNX global)
    uint32_t padding_x_right,  // pad_bottom
    uint32_t padding_y_top,    // pad_left  (ONNX global)
    uint32_t padding_y_bottom, // pad_right
    uint16_t offset_grad_in_h, uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h, uint16_t offset_grad_out_w) {

  struct blob input_blob = {0};
  struct blob coeff_blob = {0};
  struct blob output_blob = {0};

  input_blob.data = NULL;
  input_blob.diff = (float *)pGradIn;
  input_blob.W = (int)dim_im_in_y;
  input_blob.H = (int)dim_im_in_x;
  input_blob.C = (int)ch_im_in;
  input_blob.dim = (int)(ch_im_in * dim_im_in_x * dim_im_in_y);

  coeff_blob.data = (float *)pWeight;
  coeff_blob.diff = NULL;
  coeff_blob.W = (int)dim_kernel_y;
  coeff_blob.H = (int)dim_kernel_x;
  coeff_blob.C = (int)ch_im_out;
  coeff_blob.dim = (int)(ch_im_out * dim_kernel_x * dim_kernel_y);

  output_blob.data = NULL;
  output_blob.diff = (float *)pGradOut;
  output_blob.W = (int)dim_im_out_y;
  output_blob.H = (int)dim_im_out_x;
  output_blob.C = (int)ch_im_out;
  output_blob.dim = (int)(ch_im_out * dim_im_out_x * dim_im_out_y);

  struct DepthWise_Conv_args dw_args;
  memset(&dw_args, 0, sizeof(dw_args));

  dw_args.input = &input_blob;
  dw_args.coeff = &coeff_blob;
  dw_args.output = &output_blob;

  dw_args.stride_h = (int)stride_h;
  dw_args.stride_w = (int)stride_w;

  dw_args.Lpad = (int)padding_y_top;    // pad_left
  dw_args.Rpad = (int)padding_y_bottom; // pad_right
  dw_args.Upad = (int)padding_x_left;   // pad_top
  dw_args.Dpad = (int)padding_x_right;  // pad_bottom

  dw_args.offset_in_h = (int)offset_grad_in_h;
  dw_args.offset_in_w = (int)offset_grad_in_w;
  dw_args.offset_out_h = (int)offset_grad_out_h;
  dw_args.offset_out_w = (int)offset_grad_out_w;

  dw_args.skip_wg_grad = 1;
  dw_args.skip_in_grad = 0;
  dw_args.HWC = 0;

  // Dispatch calls pi_cl_team_fork internally — this is safe because
  // the binding uses ClusterTransformer (master core only).
  pulp_conv_dw_fp32_bw_input_grads_tiled_cl(&dw_args);
}
