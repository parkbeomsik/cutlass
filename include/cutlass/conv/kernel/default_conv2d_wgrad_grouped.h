/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief 
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.
  
      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/conv/kernel/implicit_gemm_convolution_grouped.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"

#include "cutlass/conv/threadblock/conv2d_wgrad_output_gradient_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_wgrad_activation_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_wgrad_output_gradient_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_wgrad_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_tile_iterator.h"

// #include "cutlass/gemm/kernel/gemm_grouped.h"
// #include "cutlass/gemm/kernel/gemm_transpose_operands.h"
// #include "cutlass/gemm/kernel/default_gemm.h"
// #include "cutlass/gemm/kernel/default_gemm_complex.h"
// #include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/layout/permute.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    typename MathOperatorTag,
    conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized,
    conv::StrideSupport StrideSupport = StrideSupport::kStrided,
    /// Access granularity of A matrix in units of elements
    int AlignmentA = 128 / cutlass::sizeof_bits<ElementA_>::value,
    /// Access granularity of B matrix in units of elements
    int AlignmentB = 128 / cutlass::sizeof_bits<ElementB_>::value,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_ = GroupScheduleMode::kDeviceOnly,
    /// Use zfill or predicate for out-of-bound cp.async
    gemm::SharedMemoryClearOption SharedMemoryClear = gemm::SharedMemoryClearOption::kNone,
    ///
    typename Enable = void
    >
struct DefaultConv2dWgradGrouped;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dWgrad specialization for Optimized IteratorAlgorithm, 
/// multi-stage pipeline, and FFMA-based mainloop for SM80

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    typename MathOperatorTag,
    conv::StrideSupport StrideSupport,
    int AccessTypeA,
    int AccessTypeB,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_,
    /// Use zfill or predicate for out-of-bound cp.async
    gemm::SharedMemoryClearOption SharedMemoryClear
>
struct DefaultConv2dWgradGrouped<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AccessTypeA,
  AccessTypeB,
  GroupScheduleMode_,
  SharedMemoryClear,
  typename platform::enable_if< ! cutlass::is_complex<ElementAccumulator>::value>::type
> {

  // Define the default GEMM kernel
  using DefaultConv2dWgradKernel = typename kernel::DefaultConv2dWgrad<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    MathOperatorTag,
    IteratorAlgorithm::kOptimized,
    StrideSupport,
    AccessTypeA,
    AccessTypeB
  >::Kernel;

    /// Define the kernel in terms of the default kernel
  using Conv2dWgradKernel = kernel::ImplicitGemmConvolutionGrouped<
    typename DefaultConv2dWgradKernel::Mma,
    typename DefaultConv2dWgradKernel::Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kWgrad,
    GroupScheduleMode_
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
