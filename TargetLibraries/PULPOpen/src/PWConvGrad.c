/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

// ============================================================================
// Pointwise Convolution Gradient Kernels
// ============================================================================

void PULP_PWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, float *__restrict__ pGradWeight) {

  struct blob input_blob = {0};
  struct blob output_blob = {0};
  struct blob coeff_blob = {0};

  // Input blob (forward activation)
  input_blob.data = (float *)pInput;
  input_blob.diff = NULL;
  input_blob.W = (int)W_in;
  input_blob.H = (int)H_in;
  input_blob.C = (int)C_in;
  input_blob.dim = (int)(C_in * H_in * W_in);

  // Output blob (gradient w.r.t. output)
  output_blob.data = NULL;
  output_blob.diff = (float *)pGradOut;
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)C_out;
  output_blob.dim = (int)(C_out * H_out * W_out);

  // Weight blob (gradient w.r.t. weights - output)
  // For PW conv: kernel is 1x1, so dim = C_out * C_in
  coeff_blob.data = NULL;
  coeff_blob.diff = (float *)pGradWeight;
  coeff_blob.W = 1;
  coeff_blob.H = 1;
  coeff_blob.C = (int)C_in;
  coeff_blob.dim = (int)(C_out * C_in);

  struct PointWise_Conv_args pw_args;
  memset(&pw_args, 0, sizeof(pw_args));

  pw_args.input = &input_blob;
  pw_args.output = &output_blob;
  pw_args.coeff = &coeff_blob;
  pw_args.transpose_buffer = NULL;

  pw_args.skip_wg_grad = 0; // Compute weight gradient
  pw_args.skip_in_grad = 1; // Skip input gradient
  pw_args.HWC = 0;          // CHW layout
  pw_args.opt_matmul_type_fw = 0;
  pw_args.opt_matmul_type_wg = 0;
  pw_args.opt_matmul_type_ig = 0;

  pulp_conv_pw_fp32_bw_param_grads_cl(&pw_args);
}

typedef struct {
  const float *pGradOut;
  const float *pWeight;
  float *pGradIn;
  uint32_t C_out;
  uint32_t C_in;
  uint32_t HW;
} pw_convgradx_args_t;

static void pw_kernel_input_grad(void *arg_) {
  const pw_convgradx_args_t *a = (const pw_convgradx_args_t *)arg_;
  const uint32_t Cin = a->C_in;
  const uint32_t Cout = a->C_out;
  const uint32_t HW = a->HW;
  const float *__restrict__ pGradOut = a->pGradOut;
  const float *__restrict__ pWeight = a->pWeight;
  float *__restrict__ pGradIn = a->pGradIn;

  // Each core owns a contiguous Cin range
  const uint32_t ci_per_core = (Cin + NUM_CORES - 1u) / NUM_CORES;
  const uint32_t ci_lo = (uint32_t)pi_core_id() * ci_per_core;
  uint32_t ci_hi = ci_lo + ci_per_core;
  if (ci_hi > Cin)
    ci_hi = Cin;
  if (ci_lo >= ci_hi)
    return;

  // Zero this core's slice of dX
  for (uint32_t ci = ci_lo; ci < ci_hi; ++ci) {
    float *dx_row = pGradIn + (size_t)ci * HW;
    for (uint32_t hw = 0; hw < HW; ++hw)
      dx_row[hw] = 0.0f;
  }

  // dX[ci, hw] = sum_co W[co, ci] * dY[co, hw]
  // Outer co loop streams a contiguous W row (length Cin) and a contiguous
  // dY row (length HW); inner axpy over hw stays contiguous on dX.
  for (uint32_t co = 0; co < Cout; ++co) {
    const float *__restrict__ w_row = pWeight + (size_t)co * Cin;
    const float *__restrict__ dy_row = pGradOut + (size_t)co * HW;
    for (uint32_t ci = ci_lo; ci < ci_hi; ++ci) {
      const float w = w_row[ci];
      float *__restrict__ dx_row = pGradIn + (size_t)ci * HW;
      for (uint32_t hw = 0; hw < HW; ++hw)
        dx_row[hw] += w * dy_row[hw];
    }
  }
}

void PULP_PWConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in) {

  // PW (1x1) has H_in == H_out and W_in == W_out at stride=1, pad=0.
  (void)H_in;
  (void)W_in;

  // RW: keep the fork-args struct off the controller's stack. The GAP9
  // pi_cl_team_fork dispatch can clobber a stack-resident args struct
  // (frame-layout-dependent), corrupting a pointer field before the worker
  // cores read it (observed: pGradIn read as garbage -> wild write). Static
  // storage is safe because cluster forks run one-at-a-time (barrier).
  static pw_convgradx_args_t args;
  args = (pw_convgradx_args_t){
      .pGradOut = pGradOut,
      .pWeight = pWeight,
      .pGradIn = pGradIn,
      .C_out = C_out,
      .C_in = C_in,
      .HW = H_out * W_out,
  };
  pi_cl_team_fork(NUM_CORES, pw_kernel_input_grad, &args);
}
