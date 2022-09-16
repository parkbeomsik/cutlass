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
/*! 
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and 
    batched array variants.
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/trace.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM Grouped
template <typename GemmKernel_>
class GemmFlexGrouped {
public:

  using GemmKernel = GemmKernel_;
  
  using ElementA = typename GemmKernel::ElementA;
  using LayoutA = typename GemmKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = GemmKernel::kTransformA;
  static int const kAlignmentA = GemmKernel::kAlignmentA;

  using ElementB = typename GemmKernel::ElementB;
  using LayoutB = typename GemmKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = GemmKernel::kTransformB;
  static int const kAlignmentB = GemmKernel::kAlignmentB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename GemmKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  static int const kAlignmentC = GemmKernel::kAlignmentC;

  using ElementAccumulator = typename GemmKernel::Mma0::Policy::Operator::ElementC;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;

  using Operator = typename GemmKernel::Operator;
  using WarpMmaOperator = typename GemmKernel::Mma0::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename WarpMmaOperator::MathOperator;
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;
   
    using ThreadblockShape0 = typename GemmKernel::Mma0::Shape;
    using WarpShape0 = typename GemmKernel::WarpShape0;
    using InstructionShape0 = typename GemmKernel::InstructionShape0;
 
    using ThreadblockShape1 = typename GemmKernel::Mma1::Shape;
    using WarpShape1 = typename GemmKernel::WarpShape1;
    using InstructionShape1 = typename GemmKernel::InstructionShape1;

  static int const kStages = GemmKernel::Mma0::kStages;

  // SplitK GEMM kernel
  using ConvertScaledOp = cutlass::epilogue::thread::Convert<
        ElementAccumulator,
        EpilogueOutputOp::kCount,
        ElementAccumulator>;

  using GemmKernelSplitK = typename kernel::DefaultGemmSplitKParallel<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape0,
    ConvertScaledOp,
    ThreadblockSwizzle,
    kStages,
    Operator
  >::GemmKernel;

  /// Reduction kernel
  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
        ElementAccumulator, typename EpilogueOutputOp::ElementAccumulator,
        EpilogueOutputOp::kCount>;

  using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
  >;

  /// Argument structure
  using Arguments = typename GemmKernel::Arguments;

protected:

  /// Kernel parameters object
  typename GemmKernel::Params params_;

  /// Reduction kernel parameters object
  typename ReductionKernel::Params reduction_params_;

public:

  /// Constructs the GEMM.
  GemmSplitKGrouped() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {
    
    return GemmKernel::can_implement(args);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
 
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape;
    size_t total_workspace_size = 0;

    for (int i = 0; i < args.problem_size; i++) {
      grid_shape = threadblock_swizzle.get_tiled_shape(
        args.problem_sizes[i], 
        {ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK},
        args.split_k_slices);
      
      total_workspace_size += sizeof(ElementAccumulator) * size_t(args.problem_sizes[i].m()) * size_t(args.problem_sizes[i].n()) * grid_shape.k();
    }

    return total_workspace_size;
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args) {

    return dim3(args.threadblock_count, 1, 1);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) {

    CUTLASS_TRACE_HOST("GemmUniversalBase::maximum_active_blocks()");

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    CUTLASS_TRACE_HOST("  smem_size: " << smem_size << " bytes");

    cudaError_t result;
    if (smem_size > (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        CUTLASS_TRACE_HOST(
          "  cudaFuncSetAttribute() returned error "
          << cudaGetErrorString(result));
        return -1;
      }
    }

    int max_active_blocks = -1;
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        Kernel<GemmKernel>,
        GemmKernel::kThreadCount0,
        smem_size);

    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST(
        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error "
        << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    CUTLASS_TRACE_HOST("GemmUniversalBase::initialize() - workspace " 
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Workspace
    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shapes;
    for (int i = 0; i < args.problem_count; i++){
      grid_shape[i] = threadblock_swizzle.get_tiled_shape(
        args.problem_size, 
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.split_k_slices);
    }

    // Define a reference to the workspace - this is an aligned region in device memory.
    TensorRef<ElementAccumulator, layout::RowMajor> * ref_workspaces =
      (TensorRef<ElementAccumulator, layout::RowMajor>*)malloc(args.problem_count * sizeof(TensorRef<ElementAccumulator, layout::RowMajor>));
    int64_t * partition_strides = (int64_t*)malloc(args.problem_count * sizeof(int64_t));

    for (int i=0; i < args.problem_count; i++) {
      ref_workspaces[i] = TensorRef<ElementAccumulator_, layout::RowMajor>(
        static_cast<ElementAccumulator_ *>(workspace), 
        args.problem_size.n());

      int64_t partition_strides[i] = int64_t(args.problem_size.m()) * int64_t(args.problem_size.n());
    }

    // Initialize the Params structure
    params_ = typename GemmKernel::Params(args, workspace);

    splitk_gemm_params_ = typename GemmKernelSplitK::Params{
      args.problem_sizes,
      grid_shapes,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      ref_workspaces,
      args.convert,
      partition_strides
    };

    reduction_params_ = typename ReductionKernel::Params(
      args.problem_sizes,
      grid_shapes,
      partition_strides,
      ref_workspaces,
      args.ref_D,
      args.ref_C.non_const_ref(),
      args.epilogue
    );
   
    // Specify shared memory capacity for kernel. 
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }
    
    params_.update(args, workspace);
    
    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    //
    // Configure grid and block dimensions
    //

    if (!params_.problem_visitor.problem_count) {
      return Status::kSuccess;
    }

    dim3 grid(params_.threadblock_count, 1, 1);
    dim3 block(GemmKernel::kThreadCount0, 1, 1);

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    //
    // Launch kernel
    //

    // Launch
    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

    //
    // Query for errors
    //
    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST("  grid launch failed with error " << cudaGetErrorString(result));
      return Status::kErrorInternal;
    }

    block = ReductionKernel::block_shape();
    grid = ReductionKernel::grid_shape(splitk_gemm_params_.problem_sizes);

    cutlass::Kernel<ReductionKernel><<< grid, block, 0, stream >>>(reduction_params_);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return Status::kErrorInternal;
    }

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
