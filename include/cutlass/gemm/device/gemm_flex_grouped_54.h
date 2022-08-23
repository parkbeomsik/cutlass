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
 
    using ThreadblockShape2 = typename GemmKernel::Mma2::Shape;
    using WarpShape2 = typename GemmKernel::WarpShape2;
    using InstructionShape2 = typename GemmKernel::InstructionShape2;
 
    using ThreadblockShape3 = typename GemmKernel::Mma3::Shape;
    using WarpShape3 = typename GemmKernel::WarpShape3;
    using InstructionShape3 = typename GemmKernel::InstructionShape3;
 
    using ThreadblockShape4 = typename GemmKernel::Mma4::Shape;
    using WarpShape4 = typename GemmKernel::WarpShape4;
    using InstructionShape4 = typename GemmKernel::InstructionShape4;
 
    using ThreadblockShape5 = typename GemmKernel::Mma5::Shape;
    using WarpShape5 = typename GemmKernel::WarpShape5;
    using InstructionShape5 = typename GemmKernel::InstructionShape5;
 
    using ThreadblockShape6 = typename GemmKernel::Mma6::Shape;
    using WarpShape6 = typename GemmKernel::WarpShape6;
    using InstructionShape6 = typename GemmKernel::InstructionShape6;
 
    using ThreadblockShape7 = typename GemmKernel::Mma7::Shape;
    using WarpShape7 = typename GemmKernel::WarpShape7;
    using InstructionShape7 = typename GemmKernel::InstructionShape7;
 
    using ThreadblockShape8 = typename GemmKernel::Mma8::Shape;
    using WarpShape8 = typename GemmKernel::WarpShape8;
    using InstructionShape8 = typename GemmKernel::InstructionShape8;
 
    using ThreadblockShape9 = typename GemmKernel::Mma9::Shape;
    using WarpShape9 = typename GemmKernel::WarpShape9;
    using InstructionShape9 = typename GemmKernel::InstructionShape9;
 
    using ThreadblockShape10 = typename GemmKernel::Mma10::Shape;
    using WarpShape10 = typename GemmKernel::WarpShape10;
    using InstructionShape10 = typename GemmKernel::InstructionShape10;
 
    using ThreadblockShape11 = typename GemmKernel::Mma11::Shape;
    using WarpShape11 = typename GemmKernel::WarpShape11;
    using InstructionShape11 = typename GemmKernel::InstructionShape11;
 
    using ThreadblockShape12 = typename GemmKernel::Mma12::Shape;
    using WarpShape12 = typename GemmKernel::WarpShape12;
    using InstructionShape12 = typename GemmKernel::InstructionShape12;
 
    using ThreadblockShape13 = typename GemmKernel::Mma13::Shape;
    using WarpShape13 = typename GemmKernel::WarpShape13;
    using InstructionShape13 = typename GemmKernel::InstructionShape13;
 
    using ThreadblockShape14 = typename GemmKernel::Mma14::Shape;
    using WarpShape14 = typename GemmKernel::WarpShape14;
    using InstructionShape14 = typename GemmKernel::InstructionShape14;
 
    using ThreadblockShape15 = typename GemmKernel::Mma15::Shape;
    using WarpShape15 = typename GemmKernel::WarpShape15;
    using InstructionShape15 = typename GemmKernel::InstructionShape15;
 
    using ThreadblockShape16 = typename GemmKernel::Mma16::Shape;
    using WarpShape16 = typename GemmKernel::WarpShape16;
    using InstructionShape16 = typename GemmKernel::InstructionShape16;
 
    using ThreadblockShape17 = typename GemmKernel::Mma17::Shape;
    using WarpShape17 = typename GemmKernel::WarpShape17;
    using InstructionShape17 = typename GemmKernel::InstructionShape17;
 
    using ThreadblockShape18 = typename GemmKernel::Mma18::Shape;
    using WarpShape18 = typename GemmKernel::WarpShape18;
    using InstructionShape18 = typename GemmKernel::InstructionShape18;
 
    using ThreadblockShape19 = typename GemmKernel::Mma19::Shape;
    using WarpShape19 = typename GemmKernel::WarpShape19;
    using InstructionShape19 = typename GemmKernel::InstructionShape19;
 
    using ThreadblockShape20 = typename GemmKernel::Mma20::Shape;
    using WarpShape20 = typename GemmKernel::WarpShape20;
    using InstructionShape20 = typename GemmKernel::InstructionShape20;
 
    using ThreadblockShape21 = typename GemmKernel::Mma21::Shape;
    using WarpShape21 = typename GemmKernel::WarpShape21;
    using InstructionShape21 = typename GemmKernel::InstructionShape21;
 
    using ThreadblockShape22 = typename GemmKernel::Mma22::Shape;
    using WarpShape22 = typename GemmKernel::WarpShape22;
    using InstructionShape22 = typename GemmKernel::InstructionShape22;
 
    using ThreadblockShape23 = typename GemmKernel::Mma23::Shape;
    using WarpShape23 = typename GemmKernel::WarpShape23;
    using InstructionShape23 = typename GemmKernel::InstructionShape23;
 
    using ThreadblockShape24 = typename GemmKernel::Mma24::Shape;
    using WarpShape24 = typename GemmKernel::WarpShape24;
    using InstructionShape24 = typename GemmKernel::InstructionShape24;
 
    using ThreadblockShape25 = typename GemmKernel::Mma25::Shape;
    using WarpShape25 = typename GemmKernel::WarpShape25;
    using InstructionShape25 = typename GemmKernel::InstructionShape25;
 
    using ThreadblockShape26 = typename GemmKernel::Mma26::Shape;
    using WarpShape26 = typename GemmKernel::WarpShape26;
    using InstructionShape26 = typename GemmKernel::InstructionShape26;
 
    using ThreadblockShape27 = typename GemmKernel::Mma27::Shape;
    using WarpShape27 = typename GemmKernel::WarpShape27;
    using InstructionShape27 = typename GemmKernel::InstructionShape27;
 
    using ThreadblockShape28 = typename GemmKernel::Mma28::Shape;
    using WarpShape28 = typename GemmKernel::WarpShape28;
    using InstructionShape28 = typename GemmKernel::InstructionShape28;
 
    using ThreadblockShape29 = typename GemmKernel::Mma29::Shape;
    using WarpShape29 = typename GemmKernel::WarpShape29;
    using InstructionShape29 = typename GemmKernel::InstructionShape29;
 
    using ThreadblockShape30 = typename GemmKernel::Mma30::Shape;
    using WarpShape30 = typename GemmKernel::WarpShape30;
    using InstructionShape30 = typename GemmKernel::InstructionShape30;
 
    using ThreadblockShape31 = typename GemmKernel::Mma31::Shape;
    using WarpShape31 = typename GemmKernel::WarpShape31;
    using InstructionShape31 = typename GemmKernel::InstructionShape31;
 
    using ThreadblockShape32 = typename GemmKernel::Mma32::Shape;
    using WarpShape32 = typename GemmKernel::WarpShape32;
    using InstructionShape32 = typename GemmKernel::InstructionShape32;
 
    using ThreadblockShape33 = typename GemmKernel::Mma33::Shape;
    using WarpShape33 = typename GemmKernel::WarpShape33;
    using InstructionShape33 = typename GemmKernel::InstructionShape33;
 
    using ThreadblockShape34 = typename GemmKernel::Mma34::Shape;
    using WarpShape34 = typename GemmKernel::WarpShape34;
    using InstructionShape34 = typename GemmKernel::InstructionShape34;
 
    using ThreadblockShape35 = typename GemmKernel::Mma35::Shape;
    using WarpShape35 = typename GemmKernel::WarpShape35;
    using InstructionShape35 = typename GemmKernel::InstructionShape35;
 
    using ThreadblockShape36 = typename GemmKernel::Mma36::Shape;
    using WarpShape36 = typename GemmKernel::WarpShape36;
    using InstructionShape36 = typename GemmKernel::InstructionShape36;
 
    using ThreadblockShape37 = typename GemmKernel::Mma37::Shape;
    using WarpShape37 = typename GemmKernel::WarpShape37;
    using InstructionShape37 = typename GemmKernel::InstructionShape37;
 
    using ThreadblockShape38 = typename GemmKernel::Mma38::Shape;
    using WarpShape38 = typename GemmKernel::WarpShape38;
    using InstructionShape38 = typename GemmKernel::InstructionShape38;
 
    using ThreadblockShape39 = typename GemmKernel::Mma39::Shape;
    using WarpShape39 = typename GemmKernel::WarpShape39;
    using InstructionShape39 = typename GemmKernel::InstructionShape39;
 
    using ThreadblockShape40 = typename GemmKernel::Mma40::Shape;
    using WarpShape40 = typename GemmKernel::WarpShape40;
    using InstructionShape40 = typename GemmKernel::InstructionShape40;
 
    using ThreadblockShape41 = typename GemmKernel::Mma41::Shape;
    using WarpShape41 = typename GemmKernel::WarpShape41;
    using InstructionShape41 = typename GemmKernel::InstructionShape41;
 
    using ThreadblockShape42 = typename GemmKernel::Mma42::Shape;
    using WarpShape42 = typename GemmKernel::WarpShape42;
    using InstructionShape42 = typename GemmKernel::InstructionShape42;
 
    using ThreadblockShape43 = typename GemmKernel::Mma43::Shape;
    using WarpShape43 = typename GemmKernel::WarpShape43;
    using InstructionShape43 = typename GemmKernel::InstructionShape43;
 
    using ThreadblockShape44 = typename GemmKernel::Mma44::Shape;
    using WarpShape44 = typename GemmKernel::WarpShape44;
    using InstructionShape44 = typename GemmKernel::InstructionShape44;
 
    using ThreadblockShape45 = typename GemmKernel::Mma45::Shape;
    using WarpShape45 = typename GemmKernel::WarpShape45;
    using InstructionShape45 = typename GemmKernel::InstructionShape45;
 
    using ThreadblockShape46 = typename GemmKernel::Mma46::Shape;
    using WarpShape46 = typename GemmKernel::WarpShape46;
    using InstructionShape46 = typename GemmKernel::InstructionShape46;
 
    using ThreadblockShape47 = typename GemmKernel::Mma47::Shape;
    using WarpShape47 = typename GemmKernel::WarpShape47;
    using InstructionShape47 = typename GemmKernel::InstructionShape47;
 
    using ThreadblockShape48 = typename GemmKernel::Mma48::Shape;
    using WarpShape48 = typename GemmKernel::WarpShape48;
    using InstructionShape48 = typename GemmKernel::InstructionShape48;
 
    using ThreadblockShape49 = typename GemmKernel::Mma49::Shape;
    using WarpShape49 = typename GemmKernel::WarpShape49;
    using InstructionShape49 = typename GemmKernel::InstructionShape49;
 
    using ThreadblockShape50 = typename GemmKernel::Mma50::Shape;
    using WarpShape50 = typename GemmKernel::WarpShape50;
    using InstructionShape50 = typename GemmKernel::InstructionShape50;
 
    using ThreadblockShape51 = typename GemmKernel::Mma51::Shape;
    using WarpShape51 = typename GemmKernel::WarpShape51;
    using InstructionShape51 = typename GemmKernel::InstructionShape51;
 
    using ThreadblockShape52 = typename GemmKernel::Mma52::Shape;
    using WarpShape52 = typename GemmKernel::WarpShape52;
    using InstructionShape52 = typename GemmKernel::InstructionShape52;
 
    using ThreadblockShape53 = typename GemmKernel::Mma53::Shape;
    using WarpShape53 = typename GemmKernel::WarpShape53;
    using InstructionShape53 = typename GemmKernel::InstructionShape53;

  static int const kStages = GemmKernel::Mma0::kStages;

  /// Argument structure
  using Arguments = typename GemmKernel::Arguments;

protected:

  /// Kernel parameters object
  typename GemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  GemmFlexGrouped() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {
    
    return GemmKernel::can_implement(args);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
 
    // This kerenl does not utilize a workspace
    return size_t();
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

    // Initialize the Params structure
    params_ = typename GemmKernel::Params(args, workspace);
   
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
  
    return Status::kSuccess;
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
