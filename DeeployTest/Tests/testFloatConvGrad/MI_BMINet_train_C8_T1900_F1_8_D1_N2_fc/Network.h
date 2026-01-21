
#ifndef __DEEPLOY_HEADER__
#define __DEEPLOY_HEADER__
#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunNetwork(uint32_t core_id, uint32_t numThreads);
void InitNetwork(uint32_t core_id, uint32_t numThread);

extern int8_t *DeeployNetwork_MEMORYARENA_L1;
static const uint32_t DeeployNetwork_MEMORYARENA_L1_len = 64000;
extern int8_t *DeeployNetwork_MEMORYARENA_L2;
static const uint32_t DeeployNetwork_MEMORYARENA_L2_len = 121601;
extern float32_t *DeeployNetwork_input_0;
static const uint32_t DeeployNetwork_input_0_len = 15200;
extern uint8_t *DeeployNetwork_input_1;
static const uint32_t DeeployNetwork_input_1_len = 1;
extern float32_t *DeeployNetwork_output_0;
static const uint32_t DeeployNetwork_output_0_len = 416;
extern float32_t *DeeployNetwork_output_1;
static const uint32_t DeeployNetwork_output_1_len = 2;
static const uint32_t DeeployNetwork_num_inputs = 2;
static const uint32_t DeeployNetwork_num_outputs = 2;
extern void *DeeployNetwork_inputs[2];
extern void *DeeployNetwork_outputs[2];
static const uint32_t DeeployNetwork_inputs_bytes[2] = {60800, 1};
static const uint32_t DeeployNetwork_outputs_bytes[2] = {1664, 8};
#endif
