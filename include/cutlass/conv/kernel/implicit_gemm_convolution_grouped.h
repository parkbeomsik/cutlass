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
    \brief Problem visitor for grouped GEMMs
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution_grouped_problem_visitor.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                           ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,                      ///! Epilogue
  typename ThreadblockSwizzle_,            ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  GroupScheduleMode GroupScheduleMode_ = conv::kernel::GroupScheduleMode::kDeviceOnly,    ///! Type of scheduling to perform
  typename ConvProblemSize_ = cutlass::conv::Conv2dProblemSize,  ///! Convolutional operator on 2D or 3D problem
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone   ///! Group mode
>
struct ImplicitGemmConvolutionGrouped {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static Operator const kConvolutionalOperator = ConvOperator;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion.
  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;

  /// Set output tensor C layout
  using LayoutC = LayoutA;

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;
  
  using Operator = typename Mma::Operator;
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  // Type definitions about the mainloop.
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;

  static int const kStages = Mma::kStages;
  static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm; 
  static StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  /// Check iterator A and B convolution dimension are the same and 
  // set device::ImplicitGemmConvolution::kConvDim
  static_assert(Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim, 
    "Convolution on different different dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  using ConvProblemSize = ConvProblemSize_;

  static conv::GroupMode const kGroupMode = GroupMode_;

  /// Wgrad C stride idx for implicit gemm algorithm 
  // Conv2d row-major matrix C (KxRSC) 
  // Conv3d row-major matrix C (KxTRSC)
  static int const kWgradCStrideIdx = 
    platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorCStrideIdx = 
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);

  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::OutputTileIterator::Layout, 
    TensorRefC,
    ConvOperator,
    ConvProblemSize
    >;

  using ProblemVisitor = ImplicitGemmConvolutionGroupedProblemVisitor<
                            ThreadblockShape,
                            kGroupScheduleMode,
                            kThreadCount,
                            kThreadCount,
                            false>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    ConvProblemSize *problem_sizes;
    int problem_count;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    TensorRefA * ref_A;
    TensorRefB * ref_B;
    TensorRefC * ref_C;
    TensorRefC * ref_D;

    // Only used by device-level operator
    ConvProblemSize *host_problem_sizes;

    SplitKMode split_k_mode;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(    
      ConvProblemSize *problem_sizes,
      int problem_count,
      int threadblock_count,
      typename EpilogueOutputOp::Params output_op,
      TensorRefA * ref_A,
      TensorRefB * ref_B,
      TensorRefC * ref_C,
      TensorRefC * ref_D,
      ConvProblemSize *host_problem_sizes=nullptr,
      SplitKMode split_k_mode=SplitKMode::kSerial
    ): 
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      threadblock_count(threadblock_count),
      output_op(output_op),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      ref_D(ref_D),
      host_problem_sizes(host_problem_sizes),
      split_k_mode(split_k_mode)
    {

    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    TensorRefA * ref_A;
    TensorRefB * ref_B;
    TensorRefC * ref_C;
    TensorRefC * ref_D;

    ElementA ** ptr_A = nullptr;
    ElementB ** ptr_B = nullptr;
    ElementC ** ptr_C = nullptr;
    ElementC ** ptr_D = nullptr;

    LayoutA layout_A;
    LayoutB layout_B;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      ref_A(nullptr),
      ref_B(nullptr),
      ref_C(nullptr),
      ref_D(nullptr),
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr)
    { }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args,
          void *workspace = nullptr,
          int tile_count = 0):
      problem_visitor(args.problem_sizes, args.problem_count, workspace, tile_count),
      threadblock_count(args.threadblock_count),
      output_op(args.output_op),
      ref_A(args.ref_A),
      ref_B(args.ref_B),
      ref_C(args.ref_C),
      ref_D(args.ref_D)
    { 

    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr,
      int tile_count = 0) {

      problem_visitor = typename ProblemVisitor::Params(args.problem_sizes, args.problem_count,
                                                        workspace, tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ref_A = args.ref_A;
      ref_B = args.ref_B;
      ref_C = args.ref_C;
      ref_D = args.ref_D;
    }

    CUTLASS_HOST_DEVICE
    void update_ptrs(
      void ** _ptr_A,
      void ** _ptr_B,
      void ** _ptr_C,
      void ** _ptr_D,
      void *workspace = nullptr,
      int tile_count = 0) {

      ptr_A = (ElementA **)_ptr_A;
      ptr_B = (ElementB **)_ptr_B;
      ptr_C = (ElementC **)_ptr_C;
      ptr_D = (ElementC **)_ptr_D;
    }

  };

  /// Shared memory storage structure
  struct SharedStorage {
    union {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    } kernel;

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  ImplicitGemmConvolutionGrouped() { } 

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  static size_t get_extra_workspace_size(
    Arguments const &args,
    cutlass::gemm::GemmCoord const &grid_tiled_shape) {

    return 0;
  }
 
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    //
    // These types shadow the type-level definitions and support the ability to implement
    // a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    //
    // Problem visitor.
    //
    ProblemVisitor problem_visitor(
      params.problem_visitor,
      shared_storage.problem_visitor,
      blockIdx.x);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {

      ConvProblemSize problem_size  = problem_visitor.problem_size();
      int32_t problem_idx     = problem_visitor.problem_index();
      int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

      // Params for ImplicitGemmConvolution
      // gemm::GemmCoord implicit_gemm_problem_size;

      int gemm_k_iterations;
      int gemm_k_iterations_per_channel;
      // typename EpilogueOutputOp::Params output_op;
      // SplitKMode split_k_mode;

      gemm::GemmCoord implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size));
      typename Mma::IteratorA::Params params_iterator_A(Mma::IteratorA::getParams(problem_size, params.ref_A[problem_idx].layout()));
      typename Mma::IteratorB::Params params_iterator_B(problem_size, params.ref_B[problem_idx].layout());
      typename Epilogue::OutputTileIterator::Params params_iterator_C(ConvOutputIteratorParameter::layout(params.ref_C[problem_idx]));
      typename Epilogue::OutputTileIterator::Params params_iterator_D(ConvOutputIteratorParameter::layout(params.ref_D[problem_idx]));

      gemm_k_iterations = implicit_gemm_k_iterations(
        kConvolutionalOperator,
        ThreadblockShape::kK,
        problem_size,
        kIteratorAlgorithm,
        kGroupMode,
        ThreadblockShape::kN);

      gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
          kConvolutionalOperator, ThreadblockShape::kK, problem_size, kIteratorAlgorithm);

      cutlass::gemm::GemmCoord grid_shape = problem_visitor.grid_shape(cutlass::conv::Operator::kWgrad, problem_size);

      cutlass::gemm::GemmCoord threadblock_offset(
        int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
        int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN,
        0);
      
      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
        threadblock_offset.m(),
        0,
      };

      cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_offset.n()
      };

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        params_iterator_A,
        problem_size,
        params.ptr_A ? (ElementA *)params.ptr_A[problem_idx] : params.ref_A[problem_idx].data(),
        thread_idx,
        tb_offset_A);

      typename Mma::IteratorB iterator_B(
        params_iterator_B,
        problem_size,
        params.ptr_B ? (ElementB *)params.ptr_B[problem_idx] : params.ref_B[problem_idx].data(),
        thread_idx,
        tb_offset_B);

      typename Mma::FragmentC accumulators;

      accumulators.clear();
      
      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

      int lane_idx = threadIdx.x % 32;

      //
      // Matrix multiply phase
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

      // Compute threadblock-scoped matrix multiply-add
      // int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();

      // Compute threadblock-scoped matrix multiply-add
      mma(
        gemm_k_iterations, 
        accumulators, 
        iterator_A, 
        iterator_B, 
        accumulators,
        gemm_k_iterations_per_channel);

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      ElementC *ptr_C = params.ptr_C ? params.ptr_C[problem_idx] : params.ref_C[problem_idx].data();
      ElementC *ptr_D = params.ptr_D ? params.ptr_D[problem_idx] : params.ref_D[problem_idx].data();

      // typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      // typename Epilogue::OutputTileIterator::Params params_D(layout_D);

      MatrixCoord threadblock_offset_output(
        threadblock_offset.m(),
        threadblock_offset.n()
      );

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(
        params_iterator_C,
        ptr_C,
        ConvOutputIteratorParameter::extent(problem_size),
        thread_idx,
        threadblock_offset_output
      );

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
        params_iterator_D,
        ptr_D,
        ConvOutputIteratorParameter::extent(problem_size),
        thread_idx,
        threadblock_offset_output
      );

      Epilogue epilogue(
        shared_storage.kernel.epilogue, 
        thread_idx, 
        warp_idx, 
        lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      epilogue(
        output_op, 
        iterator_D, 
        accumulators, 
        iterator_C); 

      // Next tile
      problem_visitor.advance(gridDim.x);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
