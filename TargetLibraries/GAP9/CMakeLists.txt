# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
  "src/**"
)

# RW: Include PULPOpen sources but exclude dory_mem related files
file(GLOB_RECURSE PULPOPEN_SOURCES "../PULPOpen/src/**")
list(FILTER PULPOPEN_SOURCES EXCLUDE REGEX ".*dory_mem.*")
list(APPEND SOURCES ${PULPOPEN_SOURCES})

if(NOT DEFINED ENV{GAP_SDK_HOME})
  message(FATAL_ERROR "Environment variable GAP_SDK_HOME not set.")
endif()

add_deeploy_library(deeploygap9 STATIC ${SOURCES})
target_include_directories(deeploygap9
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/inc
)

target_compile_options(deeploygap9 PUBLIC
    -DNUM_CORES=${NUM_CORES}
  )

target_compile_options(deeploygap9 PRIVATE
  -Wno-implicit-function-declaration
  -Wno-sign-conversion
  -Wno-sign-compare
  -Wno-type-limits
  -Wno-attributes
  -Wno-incompatible-pointer-types
)

target_link_libraries(deeploygap9 PUBLIC pmsis)

#RW: Link PULP-NN
#RW: Set PULP-NN version and bitwidth for pulp-nn-mixed
set(PULPNNVERSION XPULPV2)
set(PULPNNBITWIDTH 32)
add_compile_definitions(NUM_CORES=${NUM_CORES})

# RW: Create symlink to PULPOpen third_party if it doesn't exist
if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/third_party)
  file(CREATE_LINK
    ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party
    ${CMAKE_CURRENT_LIST_DIR}/third_party
    SYMBOLIC)
  message(STATUS "[Deeploy GAP9] Created symlink: third_party -> ../PULPOpen/third_party")
endif()

add_subdirectory(third_party/pulp-nn-mixed)

#RW: GCC not recognizing -Wno-typedef-redefinition defined in PULP-NN CMakelist
target_compile_options(pulp-nn-mixed PRIVATE -Wno-error)
target_link_libraries(deeploygap9 PUBLIC pulp-nn-mixed)
target_link_libraries(pulp-nn-mixed PUBLIC pmsis)

# Add pulp-trainlib integration
target_include_directories(deeploygap9 PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/include
)

# Add necessary pulp-trainlib sources for ConvGradW
file(GLOB PULP_TRAINLIB_CONV_SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv2d_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_im2col_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_dw_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_pw_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_matmul_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_train_utils_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_naive_fp32.c"
)
target_sources(deeploygap9 PRIVATE ${PULP_TRAINLIB_CONV_SOURCES})

# Fix compilation issues with pulp-trainlib sources
# Undefine USE_DMA and define float16alt to avoid conflicts
# Disable strict-aliasing warnings for type-punned pointer usage
set_source_files_properties(${PULP_TRAINLIB_CONV_SOURCES}
  PROPERTIES
  COMPILE_FLAGS "-UUSE_DMA -Dfloat16alt=float -fno-strict-aliasing"
)

target_link_libraries(deeploygap9 PUBLIC m)

