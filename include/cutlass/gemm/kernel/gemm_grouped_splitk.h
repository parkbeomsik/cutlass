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
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visitor class to abstract away the algorithm for iterating over tiles
template <bool Transposed = false>
struct GemmGroupedSplitKProblemVisitor {

  static bool const kTransposed = Transposed;

  struct Params {
    cutlass::gemm::GemmCoord const *problem_sizes;
    int32_t                         problem_count;

    //
    // Methods
    // 

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(): problem_sizes(nullptr), problem_count(0) { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const *problem_sizes,
      int32_t                         problem_count
    ):
      problem_sizes(problem_sizes),
      problem_count(problem_count)
    {}

  };

  struct SharedStorage {
    //
    // Nothing for now. As an optimization step, we could consider parallel
    // argmin or prefix sums across the block.
    //
  };

  //
  // Data members
  //
  
  Params const &params;
  SharedStorage &shared_storage;
  cutlass::MatrixCoord threadblock_shape0;
  cutlass::MatrixCoord threadblock_shape1;


  int64_t tile_idx;
  int64_t tile_count_sum;
  int64_t problem_tile_start;
  int32_t problem_idx;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GemmFlexGroupedProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_, 
    cutlass::MatrixCoord threadblock_shape_0,
    cutlass::MatrixCoord threadblock_shape_1,

    int32_t block_idx
  ):
    shared_storage(shared_storage_),
    params(params_),
    threadblock_shape0(threadblock_shape_0),
    threadblock_shape1(threadblock_shape_1),

    tile_idx(block_idx),
    tile_count_sum(0),
    problem_idx(0)
  {

    cutlass::gemm::GemmCoord problem = problem_size();
    cutlass::gemm::GemmCoord  grid = grid_shape(problem);

    problem_tile_start = 0;
    tile_count_sum = grid.m() * grid.n();
  }

  /// Get the grid shape
  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(
    cutlass::gemm::GemmCoord problem,
    cutlass::MatrixCoord const & block_shape) {

    return cutlass::gemm::GemmCoord(
      ((problem.m() - 1 + block_shape.row()) / block_shape.row()),
      ((problem.n() - 1 + block_shape.column()) / block_shape.column()),
      1);
  }

  /// Get the grid shape
  CUTLASS_DEVICE
  cutlass::gemm::GemmCoord grid_shape(cutlass::gemm::GemmCoord const &problem) const {
    cutlass::MatrixCoord cur_threadblock_shape;
    switch(problem_idx) {

      case 0:
        cur_threadblock_shape = threadblock_shape0;
        break;

      case 1:
        cur_threadblock_shape = threadblock_shape1;
        break;

    }
    return grid_shape(problem, cur_threadblock_shape);
  }

  /// Returns true if there is a tile to compute
  CUTLASS_DEVICE
  bool next_tile() {

    if (tile_idx < tile_count_sum) {
      return true;
    }

    do {
      ++problem_idx;

      if (problem_idx >= params.problem_count) {
        return false;
      }

      cutlass::gemm::GemmCoord problem = problem_size();
      cutlass::gemm::GemmCoord  grid = grid_shape(problem);

      int64_t tile_count = grid.m() * grid.n();

      problem_tile_start = tile_count_sum;
      tile_count_sum += tile_count;

    } while (tile_count_sum <= tile_idx);

    return true;
  }

  /// Gets the global tile index
  CUTLASS_HOST_DEVICE
  int64_t tile_index() const {
    return tile_idx;
  }

  /// Gets the index of the problem
  CUTLASS_HOST_DEVICE
  int32_t problem_index() const {
    return problem_idx;
  }

  /// Returns the problem size for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size() const {
    GemmCoord problem = params.problem_sizes[problem_idx];

    if (kTransposed) {
      swap(problem.m(), problem.n());
    }

    return problem;
  }

  CUTLASS_HOST_DEVICE
  int64_t threadblock_index() const {
    return tile_idx - problem_tile_start;
  }

  CUTLASS_DEVICE
  void advance(int32_t grid_size) {
    tile_idx += grid_size; 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma0_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma1_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue0_,             ///! Epilogue
  typename Epilogue1_,             ///! Epilogue

  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool Transposed = false
>
struct GemmGroupedSplitK {
public:

  using Mma0 = Mma0_;
  using Mma1 = Mma1_;
  using Epilogue0 = Epilogue0_;
  using Epilogue1 = Epilogue1_;

  using EpilogueOutputOp = typename Epilogue0::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kTransposed = Transposed;

  // Optional transpose (common for all shapes)
  using MapArguments = kernel::detail::MapArguments<
    typename Mma0::IteratorA::Element,
    typename Mma0::IteratorA::Layout,
    Mma0::kTransformA,
    Mma0::IteratorA::AccessType::kElements,
    typename Mma0::IteratorB::Element,
    typename Mma0::IteratorB::Layout,
    Mma0::kTransformB,
    Mma0::IteratorB::AccessType::kElements,
    typename Mma0::LayoutC,
    kTransposed
  >;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion.
  // (common for all shapes)
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue0::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator = typename Mma0::Operator;
  using OperatorClass = typename Mma0::Operator::OperatorClass;

  using ThreadblockShape0 = typename Mma0::Shape;
  using WarpShape0 = typename Mma0::Operator::Shape;
  using InstructionShape0 = typename Mma0::Policy::Operator::InstructionShape;

  using ThreadblockShape1 = typename Mma1::Shape;
  using WarpShape1 = typename Mma1::Operator::Shape;
  using InstructionShape1 = typename Mma1::Policy::Operator::InstructionShape;

  using ArchTag = typename Mma0::ArchTag;

  static int const kStages = Mma0::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC = Epilogue0::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)

  using WarpCount0 = typename Mma0::WarpCount;
  static int const kThreadCount0 = 32 * WarpCount0::kCount;

  using WarpCount1 = typename Mma1::WarpCount;
  static int const kThreadCount1 = 32 * WarpCount1::kCount;


  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord *problem_sizes;
    int problem_count;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA ** ptr_A;
    ElementB ** ptr_B;
    ElementC ** ptr_C;
    ElementC       ** ptr_D;

    typename LayoutA::Stride::LongIndex *lda;
    typename LayoutB::Stride::LongIndex *ldb;
    typename LayoutC::Stride::LongIndex *ldc;
    typename LayoutC::Stride::LongIndex *ldd;

    int split_k_slices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): 
      problem_count(0), 
      threadblock_count(0), 
      ptr_A(nullptr), 
      ptr_B(nullptr), 
      ptr_C(nullptr), 
      ptr_D(nullptr), 
      lda(nullptr),
      ldb(nullptr),
      ldc(nullptr),
      ldd(nullptr)
    {

    }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(    
      GemmCoord *problem_sizes,
      int problem_count,
      int threadblock_count,
      typename EpilogueOutputOp::Params output_op,
      ElementA ** ptr_A,
      ElementB ** ptr_B,
      ElementC ** ptr_C,
      ElementC       ** ptr_D,
      typename LayoutA::Stride::LongIndex *lda,
      typename LayoutB::Stride::LongIndex *ldb,
      typename LayoutC::Stride::LongIndex *ldc,
      typename LayoutC::Stride::LongIndex *ldd,
      int split_k_slices = 1,
    ): 
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      threadblock_count(threadblock_count),
      output_op(output_op),
      ptr_A(ptr_A),
      ptr_B(ptr_B),
      ptr_C(ptr_C),
      ptr_D(ptr_D),
      lda(lda),
      ldb(ldb),
      ldc(ldc),
      ldd(ldd),
      split_k_slices(split_k_slices)
    {

    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    typename GemmFlexGroupedProblemVisitor<kTransposed>::Params problem_visitor;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA ** ptr_A;
    ElementB ** ptr_B;
    ElementC ** ptr_C;
    ElementC ** ptr_D;

    typename LayoutA::Stride::LongIndex *lda;
    typename LayoutB::Stride::LongIndex *ldb;
    typename LayoutC::Stride::LongIndex *ldc;
    typename LayoutC::Stride::LongIndex *ldd;


    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr),
      lda(nullptr),
      ldb(nullptr),
      ldc(nullptr),
      ldd(nullptr)
    { }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args, void *workspace = nullptr):
      problem_visitor(args.problem_sizes, args.problem_count),
      threadblock_count(args.threadblock_count),
      output_op(args.output_op),
      ptr_A(args.ptr_A),
      ptr_B(args.ptr_B),
      ptr_C(args.ptr_C),
      ptr_D(args.ptr_D),
      lda(args.lda),
      ldb(args.ldb),
      ldc(args.ldc),
      ldd(args.ldd)
    { 

    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr) {

      problem_visitor = typename GemmFlexGroupedProblemVisitor<kTransposed>::Params(args.problem_sizes, args.problem_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      lda = args.lda;
      ldb = args.ldb;
      ldc = args.ldc;
      ldd = args.ldd;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename GemmFlexGroupedProblemVisitor<kTransposed>::SharedStorage problem_visitor;

    typename Mma0::SharedStorage main_loop0;
    typename Epilogue0::SharedStorage epilogue0;

    typename Mma1::SharedStorage main_loop1;
    typename Epilogue1::SharedStorage epilogue1;

  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmFlexGrouped() { } 

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
    using ElementA = typename Mma0::IteratorA::Element;
    using LayoutA = typename Mma0::IteratorA::Layout;
    using ElementB = typename Mma0::IteratorB::Element;
    using LayoutB = typename Mma0::IteratorB::Layout;
    using ElementC = typename Epilogue0::OutputTileIterator::Element;
    using LayoutC = typename Epilogue0::OutputTileIterator::Layout;

    //
    // Problem visitor.
    //
    GemmFlexGroupedProblemVisitor<kTransposed> problem_visitor(
      params.problem_visitor, 
      shared_storage.problem_visitor, 
      {Mma0::Shape::kM, Mma0::Shape::kN}, 
      {Mma1::Shape::kM, Mma1::Shape::kN}, 

      blockIdx.x);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {

      GemmCoord problem_size = problem_visitor.problem_size();
      int32_t problem_idx    = problem_visitor.problem_index();
      int32_t cta_idx        = int32_t(problem_visitor.threadblock_index());

      switch (problem_idx) {

        case 0:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma0::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma0::Shape::kN,
            0);

          // Load element pointers. Exchange pointers and strides if working on the transpose
          ElementA *ptr_A = reinterpret_cast<ElementA *>((kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
          typename LayoutA::LongIndex ldm_A = (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

          ElementB *ptr_B = reinterpret_cast<ElementB *>((kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
          typename LayoutB::LongIndex ldm_B = (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

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
          typename Mma0::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma0::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma0::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma0 mma(shared_storage.main_loop0, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma0::Shape::kK - 1) / Mma0::Shape::kK;
          

          // Wait for all threads to finish their epilogue phases from the previous tile.
          __syncthreads();

          // Compute threadblock-scoped matrix multiply-add
          mma(
            gemm_k_iterations, 
            accumulators, 
            iterator_A, 
            iterator_B, 
            accumulators);

          //
          // Epilogue
          //

          EpilogueOutputOp output_op(params.output_op);

          ElementC *ptr_C = params.ptr_C[problem_idx];
          ElementC *ptr_D = params.ptr_D[problem_idx];

          LayoutC layout_C(params.ldc[problem_idx]);
          LayoutC layout_D(params.ldd[problem_idx]);

          typename Epilogue0::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue0::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue0::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue0::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue0 epilogue(
            shared_storage.epilogue0, 
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
          break;
        }

        case 1:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma1::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma1::Shape::kN,
            0);

          // Load element pointers. Exchange pointers and strides if working on the transpose
          ElementA *ptr_A = reinterpret_cast<ElementA *>((kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
          typename LayoutA::LongIndex ldm_A = (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

          ElementB *ptr_B = reinterpret_cast<ElementB *>((kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
          typename LayoutB::LongIndex ldm_B = (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

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
          typename Mma1::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma1::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma1::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma1 mma(shared_storage.main_loop1, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma1::Shape::kK - 1) / Mma1::Shape::kK;
          

          // Wait for all threads to finish their epilogue phases from the previous tile.
          __syncthreads();

          // Compute threadblock-scoped matrix multiply-add
          mma(
            gemm_k_iterations, 
            accumulators, 
            iterator_A, 
            iterator_B, 
            accumulators);

          //
          // Epilogue
          //

          EpilogueOutputOp output_op(params.output_op);

          ElementC *ptr_C = params.ptr_C[problem_idx];
          ElementC *ptr_D = params.ptr_D[problem_idx];

          LayoutC layout_C(params.ldc[problem_idx]);
          LayoutC layout_D(params.ldd[problem_idx]);

          typename Epilogue1::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue1::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue1::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue1::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue1 epilogue(
            shared_storage.epilogue1, 
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
          break;
        }

      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

