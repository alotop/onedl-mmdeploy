// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_GRID_SAMPLER_KERNEL_HPP
#define TRT_GRID_SAMPLER_KERNEL_HPP
#include <cuda_runtime.h>

#include <cstdint>

enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };

#if NV_TENSORRT_MAJOR < 10
template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int *output_dims, int *input_dims,
                 int *grid_dims, int nb_dims, GridSamplerInterpolation interp,
                 GridSamplerPadding padding, bool align_corners, cudaStream_t stream);
#else
// TensorRT >= 10 describes dimensions with int64_t (nvinfer1::Dims::d).
// `long int` must NOT be used here: it is 32 bit on Windows (LLP64), which
// makes the plugin interpret the 64 bit dims as 32 bit values and launch a
// zero-sized grid (cudaErrorInvalidConfiguration).
template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int64_t *output_dims,
                 int64_t *input_dims, int64_t *grid_dims, int nb_dims,
                 GridSamplerInterpolation interp, GridSamplerPadding padding, bool align_corners,
                 cudaStream_t stream);
#endif  // NV_TENSORRT_MAJOR
#endif  // TRT_GRID_SAMPLER_KERNEL_HPP
