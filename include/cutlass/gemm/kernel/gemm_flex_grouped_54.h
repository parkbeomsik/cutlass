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
struct GemmFlexGroupedProblemVisitor {

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
  cutlass::MatrixCoord threadblock_shape2;
  cutlass::MatrixCoord threadblock_shape3;
  cutlass::MatrixCoord threadblock_shape4;
  cutlass::MatrixCoord threadblock_shape5;
  cutlass::MatrixCoord threadblock_shape6;
  cutlass::MatrixCoord threadblock_shape7;
  cutlass::MatrixCoord threadblock_shape8;
  cutlass::MatrixCoord threadblock_shape9;
  cutlass::MatrixCoord threadblock_shape10;
  cutlass::MatrixCoord threadblock_shape11;
  cutlass::MatrixCoord threadblock_shape12;
  cutlass::MatrixCoord threadblock_shape13;
  cutlass::MatrixCoord threadblock_shape14;
  cutlass::MatrixCoord threadblock_shape15;
  cutlass::MatrixCoord threadblock_shape16;
  cutlass::MatrixCoord threadblock_shape17;
  cutlass::MatrixCoord threadblock_shape18;
  cutlass::MatrixCoord threadblock_shape19;
  cutlass::MatrixCoord threadblock_shape20;
  cutlass::MatrixCoord threadblock_shape21;
  cutlass::MatrixCoord threadblock_shape22;
  cutlass::MatrixCoord threadblock_shape23;
  cutlass::MatrixCoord threadblock_shape24;
  cutlass::MatrixCoord threadblock_shape25;
  cutlass::MatrixCoord threadblock_shape26;
  cutlass::MatrixCoord threadblock_shape27;
  cutlass::MatrixCoord threadblock_shape28;
  cutlass::MatrixCoord threadblock_shape29;
  cutlass::MatrixCoord threadblock_shape30;
  cutlass::MatrixCoord threadblock_shape31;
  cutlass::MatrixCoord threadblock_shape32;
  cutlass::MatrixCoord threadblock_shape33;
  cutlass::MatrixCoord threadblock_shape34;
  cutlass::MatrixCoord threadblock_shape35;
  cutlass::MatrixCoord threadblock_shape36;
  cutlass::MatrixCoord threadblock_shape37;
  cutlass::MatrixCoord threadblock_shape38;
  cutlass::MatrixCoord threadblock_shape39;
  cutlass::MatrixCoord threadblock_shape40;
  cutlass::MatrixCoord threadblock_shape41;
  cutlass::MatrixCoord threadblock_shape42;
  cutlass::MatrixCoord threadblock_shape43;
  cutlass::MatrixCoord threadblock_shape44;
  cutlass::MatrixCoord threadblock_shape45;
  cutlass::MatrixCoord threadblock_shape46;
  cutlass::MatrixCoord threadblock_shape47;
  cutlass::MatrixCoord threadblock_shape48;
  cutlass::MatrixCoord threadblock_shape49;
  cutlass::MatrixCoord threadblock_shape50;
  cutlass::MatrixCoord threadblock_shape51;
  cutlass::MatrixCoord threadblock_shape52;
  cutlass::MatrixCoord threadblock_shape53;


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
    cutlass::MatrixCoord threadblock_shape_2,
    cutlass::MatrixCoord threadblock_shape_3,
    cutlass::MatrixCoord threadblock_shape_4,
    cutlass::MatrixCoord threadblock_shape_5,
    cutlass::MatrixCoord threadblock_shape_6,
    cutlass::MatrixCoord threadblock_shape_7,
    cutlass::MatrixCoord threadblock_shape_8,
    cutlass::MatrixCoord threadblock_shape_9,
    cutlass::MatrixCoord threadblock_shape_10,
    cutlass::MatrixCoord threadblock_shape_11,
    cutlass::MatrixCoord threadblock_shape_12,
    cutlass::MatrixCoord threadblock_shape_13,
    cutlass::MatrixCoord threadblock_shape_14,
    cutlass::MatrixCoord threadblock_shape_15,
    cutlass::MatrixCoord threadblock_shape_16,
    cutlass::MatrixCoord threadblock_shape_17,
    cutlass::MatrixCoord threadblock_shape_18,
    cutlass::MatrixCoord threadblock_shape_19,
    cutlass::MatrixCoord threadblock_shape_20,
    cutlass::MatrixCoord threadblock_shape_21,
    cutlass::MatrixCoord threadblock_shape_22,
    cutlass::MatrixCoord threadblock_shape_23,
    cutlass::MatrixCoord threadblock_shape_24,
    cutlass::MatrixCoord threadblock_shape_25,
    cutlass::MatrixCoord threadblock_shape_26,
    cutlass::MatrixCoord threadblock_shape_27,
    cutlass::MatrixCoord threadblock_shape_28,
    cutlass::MatrixCoord threadblock_shape_29,
    cutlass::MatrixCoord threadblock_shape_30,
    cutlass::MatrixCoord threadblock_shape_31,
    cutlass::MatrixCoord threadblock_shape_32,
    cutlass::MatrixCoord threadblock_shape_33,
    cutlass::MatrixCoord threadblock_shape_34,
    cutlass::MatrixCoord threadblock_shape_35,
    cutlass::MatrixCoord threadblock_shape_36,
    cutlass::MatrixCoord threadblock_shape_37,
    cutlass::MatrixCoord threadblock_shape_38,
    cutlass::MatrixCoord threadblock_shape_39,
    cutlass::MatrixCoord threadblock_shape_40,
    cutlass::MatrixCoord threadblock_shape_41,
    cutlass::MatrixCoord threadblock_shape_42,
    cutlass::MatrixCoord threadblock_shape_43,
    cutlass::MatrixCoord threadblock_shape_44,
    cutlass::MatrixCoord threadblock_shape_45,
    cutlass::MatrixCoord threadblock_shape_46,
    cutlass::MatrixCoord threadblock_shape_47,
    cutlass::MatrixCoord threadblock_shape_48,
    cutlass::MatrixCoord threadblock_shape_49,
    cutlass::MatrixCoord threadblock_shape_50,
    cutlass::MatrixCoord threadblock_shape_51,
    cutlass::MatrixCoord threadblock_shape_52,
    cutlass::MatrixCoord threadblock_shape_53,

    int32_t block_idx
  ):
    shared_storage(shared_storage_),
    params(params_),
    threadblock_shape0(threadblock_shape_0),
    threadblock_shape1(threadblock_shape_1),
    threadblock_shape2(threadblock_shape_2),
    threadblock_shape3(threadblock_shape_3),
    threadblock_shape4(threadblock_shape_4),
    threadblock_shape5(threadblock_shape_5),
    threadblock_shape6(threadblock_shape_6),
    threadblock_shape7(threadblock_shape_7),
    threadblock_shape8(threadblock_shape_8),
    threadblock_shape9(threadblock_shape_9),
    threadblock_shape10(threadblock_shape_10),
    threadblock_shape11(threadblock_shape_11),
    threadblock_shape12(threadblock_shape_12),
    threadblock_shape13(threadblock_shape_13),
    threadblock_shape14(threadblock_shape_14),
    threadblock_shape15(threadblock_shape_15),
    threadblock_shape16(threadblock_shape_16),
    threadblock_shape17(threadblock_shape_17),
    threadblock_shape18(threadblock_shape_18),
    threadblock_shape19(threadblock_shape_19),
    threadblock_shape20(threadblock_shape_20),
    threadblock_shape21(threadblock_shape_21),
    threadblock_shape22(threadblock_shape_22),
    threadblock_shape23(threadblock_shape_23),
    threadblock_shape24(threadblock_shape_24),
    threadblock_shape25(threadblock_shape_25),
    threadblock_shape26(threadblock_shape_26),
    threadblock_shape27(threadblock_shape_27),
    threadblock_shape28(threadblock_shape_28),
    threadblock_shape29(threadblock_shape_29),
    threadblock_shape30(threadblock_shape_30),
    threadblock_shape31(threadblock_shape_31),
    threadblock_shape32(threadblock_shape_32),
    threadblock_shape33(threadblock_shape_33),
    threadblock_shape34(threadblock_shape_34),
    threadblock_shape35(threadblock_shape_35),
    threadblock_shape36(threadblock_shape_36),
    threadblock_shape37(threadblock_shape_37),
    threadblock_shape38(threadblock_shape_38),
    threadblock_shape39(threadblock_shape_39),
    threadblock_shape40(threadblock_shape_40),
    threadblock_shape41(threadblock_shape_41),
    threadblock_shape42(threadblock_shape_42),
    threadblock_shape43(threadblock_shape_43),
    threadblock_shape44(threadblock_shape_44),
    threadblock_shape45(threadblock_shape_45),
    threadblock_shape46(threadblock_shape_46),
    threadblock_shape47(threadblock_shape_47),
    threadblock_shape48(threadblock_shape_48),
    threadblock_shape49(threadblock_shape_49),
    threadblock_shape50(threadblock_shape_50),
    threadblock_shape51(threadblock_shape_51),
    threadblock_shape52(threadblock_shape_52),
    threadblock_shape53(threadblock_shape_53),

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

      case 2:
        cur_threadblock_shape = threadblock_shape2;
        break;

      case 3:
        cur_threadblock_shape = threadblock_shape3;
        break;

      case 4:
        cur_threadblock_shape = threadblock_shape4;
        break;

      case 5:
        cur_threadblock_shape = threadblock_shape5;
        break;

      case 6:
        cur_threadblock_shape = threadblock_shape6;
        break;

      case 7:
        cur_threadblock_shape = threadblock_shape7;
        break;

      case 8:
        cur_threadblock_shape = threadblock_shape8;
        break;

      case 9:
        cur_threadblock_shape = threadblock_shape9;
        break;

      case 10:
        cur_threadblock_shape = threadblock_shape10;
        break;

      case 11:
        cur_threadblock_shape = threadblock_shape11;
        break;

      case 12:
        cur_threadblock_shape = threadblock_shape12;
        break;

      case 13:
        cur_threadblock_shape = threadblock_shape13;
        break;

      case 14:
        cur_threadblock_shape = threadblock_shape14;
        break;

      case 15:
        cur_threadblock_shape = threadblock_shape15;
        break;

      case 16:
        cur_threadblock_shape = threadblock_shape16;
        break;

      case 17:
        cur_threadblock_shape = threadblock_shape17;
        break;

      case 18:
        cur_threadblock_shape = threadblock_shape18;
        break;

      case 19:
        cur_threadblock_shape = threadblock_shape19;
        break;

      case 20:
        cur_threadblock_shape = threadblock_shape20;
        break;

      case 21:
        cur_threadblock_shape = threadblock_shape21;
        break;

      case 22:
        cur_threadblock_shape = threadblock_shape22;
        break;

      case 23:
        cur_threadblock_shape = threadblock_shape23;
        break;

      case 24:
        cur_threadblock_shape = threadblock_shape24;
        break;

      case 25:
        cur_threadblock_shape = threadblock_shape25;
        break;

      case 26:
        cur_threadblock_shape = threadblock_shape26;
        break;

      case 27:
        cur_threadblock_shape = threadblock_shape27;
        break;

      case 28:
        cur_threadblock_shape = threadblock_shape28;
        break;

      case 29:
        cur_threadblock_shape = threadblock_shape29;
        break;

      case 30:
        cur_threadblock_shape = threadblock_shape30;
        break;

      case 31:
        cur_threadblock_shape = threadblock_shape31;
        break;

      case 32:
        cur_threadblock_shape = threadblock_shape32;
        break;

      case 33:
        cur_threadblock_shape = threadblock_shape33;
        break;

      case 34:
        cur_threadblock_shape = threadblock_shape34;
        break;

      case 35:
        cur_threadblock_shape = threadblock_shape35;
        break;

      case 36:
        cur_threadblock_shape = threadblock_shape36;
        break;

      case 37:
        cur_threadblock_shape = threadblock_shape37;
        break;

      case 38:
        cur_threadblock_shape = threadblock_shape38;
        break;

      case 39:
        cur_threadblock_shape = threadblock_shape39;
        break;

      case 40:
        cur_threadblock_shape = threadblock_shape40;
        break;

      case 41:
        cur_threadblock_shape = threadblock_shape41;
        break;

      case 42:
        cur_threadblock_shape = threadblock_shape42;
        break;

      case 43:
        cur_threadblock_shape = threadblock_shape43;
        break;

      case 44:
        cur_threadblock_shape = threadblock_shape44;
        break;

      case 45:
        cur_threadblock_shape = threadblock_shape45;
        break;

      case 46:
        cur_threadblock_shape = threadblock_shape46;
        break;

      case 47:
        cur_threadblock_shape = threadblock_shape47;
        break;

      case 48:
        cur_threadblock_shape = threadblock_shape48;
        break;

      case 49:
        cur_threadblock_shape = threadblock_shape49;
        break;

      case 50:
        cur_threadblock_shape = threadblock_shape50;
        break;

      case 51:
        cur_threadblock_shape = threadblock_shape51;
        break;

      case 52:
        cur_threadblock_shape = threadblock_shape52;
        break;

      case 53:
        cur_threadblock_shape = threadblock_shape53;
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
  typename Mma2_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma3_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma4_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma5_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma6_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma7_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma8_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma9_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma10_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma11_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma12_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma13_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma14_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma15_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma16_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma17_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma18_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma19_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma20_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma21_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma22_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma23_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma24_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma25_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma26_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma27_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma28_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma29_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma30_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma31_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma32_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma33_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma34_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma35_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma36_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma37_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma38_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma39_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma40_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma41_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma42_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma43_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma44_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma45_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma46_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma47_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma48_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma49_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma50_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma51_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma52_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Mma53_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue0_,             ///! Epilogue
  typename Epilogue1_,             ///! Epilogue
  typename Epilogue2_,             ///! Epilogue
  typename Epilogue3_,             ///! Epilogue
  typename Epilogue4_,             ///! Epilogue
  typename Epilogue5_,             ///! Epilogue
  typename Epilogue6_,             ///! Epilogue
  typename Epilogue7_,             ///! Epilogue
  typename Epilogue8_,             ///! Epilogue
  typename Epilogue9_,             ///! Epilogue
  typename Epilogue10_,             ///! Epilogue
  typename Epilogue11_,             ///! Epilogue
  typename Epilogue12_,             ///! Epilogue
  typename Epilogue13_,             ///! Epilogue
  typename Epilogue14_,             ///! Epilogue
  typename Epilogue15_,             ///! Epilogue
  typename Epilogue16_,             ///! Epilogue
  typename Epilogue17_,             ///! Epilogue
  typename Epilogue18_,             ///! Epilogue
  typename Epilogue19_,             ///! Epilogue
  typename Epilogue20_,             ///! Epilogue
  typename Epilogue21_,             ///! Epilogue
  typename Epilogue22_,             ///! Epilogue
  typename Epilogue23_,             ///! Epilogue
  typename Epilogue24_,             ///! Epilogue
  typename Epilogue25_,             ///! Epilogue
  typename Epilogue26_,             ///! Epilogue
  typename Epilogue27_,             ///! Epilogue
  typename Epilogue28_,             ///! Epilogue
  typename Epilogue29_,             ///! Epilogue
  typename Epilogue30_,             ///! Epilogue
  typename Epilogue31_,             ///! Epilogue
  typename Epilogue32_,             ///! Epilogue
  typename Epilogue33_,             ///! Epilogue
  typename Epilogue34_,             ///! Epilogue
  typename Epilogue35_,             ///! Epilogue
  typename Epilogue36_,             ///! Epilogue
  typename Epilogue37_,             ///! Epilogue
  typename Epilogue38_,             ///! Epilogue
  typename Epilogue39_,             ///! Epilogue
  typename Epilogue40_,             ///! Epilogue
  typename Epilogue41_,             ///! Epilogue
  typename Epilogue42_,             ///! Epilogue
  typename Epilogue43_,             ///! Epilogue
  typename Epilogue44_,             ///! Epilogue
  typename Epilogue45_,             ///! Epilogue
  typename Epilogue46_,             ///! Epilogue
  typename Epilogue47_,             ///! Epilogue
  typename Epilogue48_,             ///! Epilogue
  typename Epilogue49_,             ///! Epilogue
  typename Epilogue50_,             ///! Epilogue
  typename Epilogue51_,             ///! Epilogue
  typename Epilogue52_,             ///! Epilogue
  typename Epilogue53_,             ///! Epilogue

  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool Transposed = false
>
struct GemmFlexGrouped {
public:

  using Mma0 = Mma0_;
  using Mma1 = Mma1_;
  using Mma2 = Mma2_;
  using Mma3 = Mma3_;
  using Mma4 = Mma4_;
  using Mma5 = Mma5_;
  using Mma6 = Mma6_;
  using Mma7 = Mma7_;
  using Mma8 = Mma8_;
  using Mma9 = Mma9_;
  using Mma10 = Mma10_;
  using Mma11 = Mma11_;
  using Mma12 = Mma12_;
  using Mma13 = Mma13_;
  using Mma14 = Mma14_;
  using Mma15 = Mma15_;
  using Mma16 = Mma16_;
  using Mma17 = Mma17_;
  using Mma18 = Mma18_;
  using Mma19 = Mma19_;
  using Mma20 = Mma20_;
  using Mma21 = Mma21_;
  using Mma22 = Mma22_;
  using Mma23 = Mma23_;
  using Mma24 = Mma24_;
  using Mma25 = Mma25_;
  using Mma26 = Mma26_;
  using Mma27 = Mma27_;
  using Mma28 = Mma28_;
  using Mma29 = Mma29_;
  using Mma30 = Mma30_;
  using Mma31 = Mma31_;
  using Mma32 = Mma32_;
  using Mma33 = Mma33_;
  using Mma34 = Mma34_;
  using Mma35 = Mma35_;
  using Mma36 = Mma36_;
  using Mma37 = Mma37_;
  using Mma38 = Mma38_;
  using Mma39 = Mma39_;
  using Mma40 = Mma40_;
  using Mma41 = Mma41_;
  using Mma42 = Mma42_;
  using Mma43 = Mma43_;
  using Mma44 = Mma44_;
  using Mma45 = Mma45_;
  using Mma46 = Mma46_;
  using Mma47 = Mma47_;
  using Mma48 = Mma48_;
  using Mma49 = Mma49_;
  using Mma50 = Mma50_;
  using Mma51 = Mma51_;
  using Mma52 = Mma52_;
  using Mma53 = Mma53_;
  using Epilogue0 = Epilogue0_;
  using Epilogue1 = Epilogue1_;
  using Epilogue2 = Epilogue2_;
  using Epilogue3 = Epilogue3_;
  using Epilogue4 = Epilogue4_;
  using Epilogue5 = Epilogue5_;
  using Epilogue6 = Epilogue6_;
  using Epilogue7 = Epilogue7_;
  using Epilogue8 = Epilogue8_;
  using Epilogue9 = Epilogue9_;
  using Epilogue10 = Epilogue10_;
  using Epilogue11 = Epilogue11_;
  using Epilogue12 = Epilogue12_;
  using Epilogue13 = Epilogue13_;
  using Epilogue14 = Epilogue14_;
  using Epilogue15 = Epilogue15_;
  using Epilogue16 = Epilogue16_;
  using Epilogue17 = Epilogue17_;
  using Epilogue18 = Epilogue18_;
  using Epilogue19 = Epilogue19_;
  using Epilogue20 = Epilogue20_;
  using Epilogue21 = Epilogue21_;
  using Epilogue22 = Epilogue22_;
  using Epilogue23 = Epilogue23_;
  using Epilogue24 = Epilogue24_;
  using Epilogue25 = Epilogue25_;
  using Epilogue26 = Epilogue26_;
  using Epilogue27 = Epilogue27_;
  using Epilogue28 = Epilogue28_;
  using Epilogue29 = Epilogue29_;
  using Epilogue30 = Epilogue30_;
  using Epilogue31 = Epilogue31_;
  using Epilogue32 = Epilogue32_;
  using Epilogue33 = Epilogue33_;
  using Epilogue34 = Epilogue34_;
  using Epilogue35 = Epilogue35_;
  using Epilogue36 = Epilogue36_;
  using Epilogue37 = Epilogue37_;
  using Epilogue38 = Epilogue38_;
  using Epilogue39 = Epilogue39_;
  using Epilogue40 = Epilogue40_;
  using Epilogue41 = Epilogue41_;
  using Epilogue42 = Epilogue42_;
  using Epilogue43 = Epilogue43_;
  using Epilogue44 = Epilogue44_;
  using Epilogue45 = Epilogue45_;
  using Epilogue46 = Epilogue46_;
  using Epilogue47 = Epilogue47_;
  using Epilogue48 = Epilogue48_;
  using Epilogue49 = Epilogue49_;
  using Epilogue50 = Epilogue50_;
  using Epilogue51 = Epilogue51_;
  using Epilogue52 = Epilogue52_;
  using Epilogue53 = Epilogue53_;

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

  using ThreadblockShape2 = typename Mma2::Shape;
  using WarpShape2 = typename Mma2::Operator::Shape;
  using InstructionShape2 = typename Mma2::Policy::Operator::InstructionShape;

  using ThreadblockShape3 = typename Mma3::Shape;
  using WarpShape3 = typename Mma3::Operator::Shape;
  using InstructionShape3 = typename Mma3::Policy::Operator::InstructionShape;

  using ThreadblockShape4 = typename Mma4::Shape;
  using WarpShape4 = typename Mma4::Operator::Shape;
  using InstructionShape4 = typename Mma4::Policy::Operator::InstructionShape;

  using ThreadblockShape5 = typename Mma5::Shape;
  using WarpShape5 = typename Mma5::Operator::Shape;
  using InstructionShape5 = typename Mma5::Policy::Operator::InstructionShape;

  using ThreadblockShape6 = typename Mma6::Shape;
  using WarpShape6 = typename Mma6::Operator::Shape;
  using InstructionShape6 = typename Mma6::Policy::Operator::InstructionShape;

  using ThreadblockShape7 = typename Mma7::Shape;
  using WarpShape7 = typename Mma7::Operator::Shape;
  using InstructionShape7 = typename Mma7::Policy::Operator::InstructionShape;

  using ThreadblockShape8 = typename Mma8::Shape;
  using WarpShape8 = typename Mma8::Operator::Shape;
  using InstructionShape8 = typename Mma8::Policy::Operator::InstructionShape;

  using ThreadblockShape9 = typename Mma9::Shape;
  using WarpShape9 = typename Mma9::Operator::Shape;
  using InstructionShape9 = typename Mma9::Policy::Operator::InstructionShape;

  using ThreadblockShape10 = typename Mma10::Shape;
  using WarpShape10 = typename Mma10::Operator::Shape;
  using InstructionShape10 = typename Mma10::Policy::Operator::InstructionShape;

  using ThreadblockShape11 = typename Mma11::Shape;
  using WarpShape11 = typename Mma11::Operator::Shape;
  using InstructionShape11 = typename Mma11::Policy::Operator::InstructionShape;

  using ThreadblockShape12 = typename Mma12::Shape;
  using WarpShape12 = typename Mma12::Operator::Shape;
  using InstructionShape12 = typename Mma12::Policy::Operator::InstructionShape;

  using ThreadblockShape13 = typename Mma13::Shape;
  using WarpShape13 = typename Mma13::Operator::Shape;
  using InstructionShape13 = typename Mma13::Policy::Operator::InstructionShape;

  using ThreadblockShape14 = typename Mma14::Shape;
  using WarpShape14 = typename Mma14::Operator::Shape;
  using InstructionShape14 = typename Mma14::Policy::Operator::InstructionShape;

  using ThreadblockShape15 = typename Mma15::Shape;
  using WarpShape15 = typename Mma15::Operator::Shape;
  using InstructionShape15 = typename Mma15::Policy::Operator::InstructionShape;

  using ThreadblockShape16 = typename Mma16::Shape;
  using WarpShape16 = typename Mma16::Operator::Shape;
  using InstructionShape16 = typename Mma16::Policy::Operator::InstructionShape;

  using ThreadblockShape17 = typename Mma17::Shape;
  using WarpShape17 = typename Mma17::Operator::Shape;
  using InstructionShape17 = typename Mma17::Policy::Operator::InstructionShape;

  using ThreadblockShape18 = typename Mma18::Shape;
  using WarpShape18 = typename Mma18::Operator::Shape;
  using InstructionShape18 = typename Mma18::Policy::Operator::InstructionShape;

  using ThreadblockShape19 = typename Mma19::Shape;
  using WarpShape19 = typename Mma19::Operator::Shape;
  using InstructionShape19 = typename Mma19::Policy::Operator::InstructionShape;

  using ThreadblockShape20 = typename Mma20::Shape;
  using WarpShape20 = typename Mma20::Operator::Shape;
  using InstructionShape20 = typename Mma20::Policy::Operator::InstructionShape;

  using ThreadblockShape21 = typename Mma21::Shape;
  using WarpShape21 = typename Mma21::Operator::Shape;
  using InstructionShape21 = typename Mma21::Policy::Operator::InstructionShape;

  using ThreadblockShape22 = typename Mma22::Shape;
  using WarpShape22 = typename Mma22::Operator::Shape;
  using InstructionShape22 = typename Mma22::Policy::Operator::InstructionShape;

  using ThreadblockShape23 = typename Mma23::Shape;
  using WarpShape23 = typename Mma23::Operator::Shape;
  using InstructionShape23 = typename Mma23::Policy::Operator::InstructionShape;

  using ThreadblockShape24 = typename Mma24::Shape;
  using WarpShape24 = typename Mma24::Operator::Shape;
  using InstructionShape24 = typename Mma24::Policy::Operator::InstructionShape;

  using ThreadblockShape25 = typename Mma25::Shape;
  using WarpShape25 = typename Mma25::Operator::Shape;
  using InstructionShape25 = typename Mma25::Policy::Operator::InstructionShape;

  using ThreadblockShape26 = typename Mma26::Shape;
  using WarpShape26 = typename Mma26::Operator::Shape;
  using InstructionShape26 = typename Mma26::Policy::Operator::InstructionShape;

  using ThreadblockShape27 = typename Mma27::Shape;
  using WarpShape27 = typename Mma27::Operator::Shape;
  using InstructionShape27 = typename Mma27::Policy::Operator::InstructionShape;

  using ThreadblockShape28 = typename Mma28::Shape;
  using WarpShape28 = typename Mma28::Operator::Shape;
  using InstructionShape28 = typename Mma28::Policy::Operator::InstructionShape;

  using ThreadblockShape29 = typename Mma29::Shape;
  using WarpShape29 = typename Mma29::Operator::Shape;
  using InstructionShape29 = typename Mma29::Policy::Operator::InstructionShape;

  using ThreadblockShape30 = typename Mma30::Shape;
  using WarpShape30 = typename Mma30::Operator::Shape;
  using InstructionShape30 = typename Mma30::Policy::Operator::InstructionShape;

  using ThreadblockShape31 = typename Mma31::Shape;
  using WarpShape31 = typename Mma31::Operator::Shape;
  using InstructionShape31 = typename Mma31::Policy::Operator::InstructionShape;

  using ThreadblockShape32 = typename Mma32::Shape;
  using WarpShape32 = typename Mma32::Operator::Shape;
  using InstructionShape32 = typename Mma32::Policy::Operator::InstructionShape;

  using ThreadblockShape33 = typename Mma33::Shape;
  using WarpShape33 = typename Mma33::Operator::Shape;
  using InstructionShape33 = typename Mma33::Policy::Operator::InstructionShape;

  using ThreadblockShape34 = typename Mma34::Shape;
  using WarpShape34 = typename Mma34::Operator::Shape;
  using InstructionShape34 = typename Mma34::Policy::Operator::InstructionShape;

  using ThreadblockShape35 = typename Mma35::Shape;
  using WarpShape35 = typename Mma35::Operator::Shape;
  using InstructionShape35 = typename Mma35::Policy::Operator::InstructionShape;

  using ThreadblockShape36 = typename Mma36::Shape;
  using WarpShape36 = typename Mma36::Operator::Shape;
  using InstructionShape36 = typename Mma36::Policy::Operator::InstructionShape;

  using ThreadblockShape37 = typename Mma37::Shape;
  using WarpShape37 = typename Mma37::Operator::Shape;
  using InstructionShape37 = typename Mma37::Policy::Operator::InstructionShape;

  using ThreadblockShape38 = typename Mma38::Shape;
  using WarpShape38 = typename Mma38::Operator::Shape;
  using InstructionShape38 = typename Mma38::Policy::Operator::InstructionShape;

  using ThreadblockShape39 = typename Mma39::Shape;
  using WarpShape39 = typename Mma39::Operator::Shape;
  using InstructionShape39 = typename Mma39::Policy::Operator::InstructionShape;

  using ThreadblockShape40 = typename Mma40::Shape;
  using WarpShape40 = typename Mma40::Operator::Shape;
  using InstructionShape40 = typename Mma40::Policy::Operator::InstructionShape;

  using ThreadblockShape41 = typename Mma41::Shape;
  using WarpShape41 = typename Mma41::Operator::Shape;
  using InstructionShape41 = typename Mma41::Policy::Operator::InstructionShape;

  using ThreadblockShape42 = typename Mma42::Shape;
  using WarpShape42 = typename Mma42::Operator::Shape;
  using InstructionShape42 = typename Mma42::Policy::Operator::InstructionShape;

  using ThreadblockShape43 = typename Mma43::Shape;
  using WarpShape43 = typename Mma43::Operator::Shape;
  using InstructionShape43 = typename Mma43::Policy::Operator::InstructionShape;

  using ThreadblockShape44 = typename Mma44::Shape;
  using WarpShape44 = typename Mma44::Operator::Shape;
  using InstructionShape44 = typename Mma44::Policy::Operator::InstructionShape;

  using ThreadblockShape45 = typename Mma45::Shape;
  using WarpShape45 = typename Mma45::Operator::Shape;
  using InstructionShape45 = typename Mma45::Policy::Operator::InstructionShape;

  using ThreadblockShape46 = typename Mma46::Shape;
  using WarpShape46 = typename Mma46::Operator::Shape;
  using InstructionShape46 = typename Mma46::Policy::Operator::InstructionShape;

  using ThreadblockShape47 = typename Mma47::Shape;
  using WarpShape47 = typename Mma47::Operator::Shape;
  using InstructionShape47 = typename Mma47::Policy::Operator::InstructionShape;

  using ThreadblockShape48 = typename Mma48::Shape;
  using WarpShape48 = typename Mma48::Operator::Shape;
  using InstructionShape48 = typename Mma48::Policy::Operator::InstructionShape;

  using ThreadblockShape49 = typename Mma49::Shape;
  using WarpShape49 = typename Mma49::Operator::Shape;
  using InstructionShape49 = typename Mma49::Policy::Operator::InstructionShape;

  using ThreadblockShape50 = typename Mma50::Shape;
  using WarpShape50 = typename Mma50::Operator::Shape;
  using InstructionShape50 = typename Mma50::Policy::Operator::InstructionShape;

  using ThreadblockShape51 = typename Mma51::Shape;
  using WarpShape51 = typename Mma51::Operator::Shape;
  using InstructionShape51 = typename Mma51::Policy::Operator::InstructionShape;

  using ThreadblockShape52 = typename Mma52::Shape;
  using WarpShape52 = typename Mma52::Operator::Shape;
  using InstructionShape52 = typename Mma52::Policy::Operator::InstructionShape;

  using ThreadblockShape53 = typename Mma53::Shape;
  using WarpShape53 = typename Mma53::Operator::Shape;
  using InstructionShape53 = typename Mma53::Policy::Operator::InstructionShape;

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

  using WarpCount2 = typename Mma2::WarpCount;
  static int const kThreadCount2 = 32 * WarpCount2::kCount;

  using WarpCount3 = typename Mma3::WarpCount;
  static int const kThreadCount3 = 32 * WarpCount3::kCount;

  using WarpCount4 = typename Mma4::WarpCount;
  static int const kThreadCount4 = 32 * WarpCount4::kCount;

  using WarpCount5 = typename Mma5::WarpCount;
  static int const kThreadCount5 = 32 * WarpCount5::kCount;

  using WarpCount6 = typename Mma6::WarpCount;
  static int const kThreadCount6 = 32 * WarpCount6::kCount;

  using WarpCount7 = typename Mma7::WarpCount;
  static int const kThreadCount7 = 32 * WarpCount7::kCount;

  using WarpCount8 = typename Mma8::WarpCount;
  static int const kThreadCount8 = 32 * WarpCount8::kCount;

  using WarpCount9 = typename Mma9::WarpCount;
  static int const kThreadCount9 = 32 * WarpCount9::kCount;

  using WarpCount10 = typename Mma10::WarpCount;
  static int const kThreadCount10 = 32 * WarpCount10::kCount;

  using WarpCount11 = typename Mma11::WarpCount;
  static int const kThreadCount11 = 32 * WarpCount11::kCount;

  using WarpCount12 = typename Mma12::WarpCount;
  static int const kThreadCount12 = 32 * WarpCount12::kCount;

  using WarpCount13 = typename Mma13::WarpCount;
  static int const kThreadCount13 = 32 * WarpCount13::kCount;

  using WarpCount14 = typename Mma14::WarpCount;
  static int const kThreadCount14 = 32 * WarpCount14::kCount;

  using WarpCount15 = typename Mma15::WarpCount;
  static int const kThreadCount15 = 32 * WarpCount15::kCount;

  using WarpCount16 = typename Mma16::WarpCount;
  static int const kThreadCount16 = 32 * WarpCount16::kCount;

  using WarpCount17 = typename Mma17::WarpCount;
  static int const kThreadCount17 = 32 * WarpCount17::kCount;

  using WarpCount18 = typename Mma18::WarpCount;
  static int const kThreadCount18 = 32 * WarpCount18::kCount;

  using WarpCount19 = typename Mma19::WarpCount;
  static int const kThreadCount19 = 32 * WarpCount19::kCount;

  using WarpCount20 = typename Mma20::WarpCount;
  static int const kThreadCount20 = 32 * WarpCount20::kCount;

  using WarpCount21 = typename Mma21::WarpCount;
  static int const kThreadCount21 = 32 * WarpCount21::kCount;

  using WarpCount22 = typename Mma22::WarpCount;
  static int const kThreadCount22 = 32 * WarpCount22::kCount;

  using WarpCount23 = typename Mma23::WarpCount;
  static int const kThreadCount23 = 32 * WarpCount23::kCount;

  using WarpCount24 = typename Mma24::WarpCount;
  static int const kThreadCount24 = 32 * WarpCount24::kCount;

  using WarpCount25 = typename Mma25::WarpCount;
  static int const kThreadCount25 = 32 * WarpCount25::kCount;

  using WarpCount26 = typename Mma26::WarpCount;
  static int const kThreadCount26 = 32 * WarpCount26::kCount;

  using WarpCount27 = typename Mma27::WarpCount;
  static int const kThreadCount27 = 32 * WarpCount27::kCount;

  using WarpCount28 = typename Mma28::WarpCount;
  static int const kThreadCount28 = 32 * WarpCount28::kCount;

  using WarpCount29 = typename Mma29::WarpCount;
  static int const kThreadCount29 = 32 * WarpCount29::kCount;

  using WarpCount30 = typename Mma30::WarpCount;
  static int const kThreadCount30 = 32 * WarpCount30::kCount;

  using WarpCount31 = typename Mma31::WarpCount;
  static int const kThreadCount31 = 32 * WarpCount31::kCount;

  using WarpCount32 = typename Mma32::WarpCount;
  static int const kThreadCount32 = 32 * WarpCount32::kCount;

  using WarpCount33 = typename Mma33::WarpCount;
  static int const kThreadCount33 = 32 * WarpCount33::kCount;

  using WarpCount34 = typename Mma34::WarpCount;
  static int const kThreadCount34 = 32 * WarpCount34::kCount;

  using WarpCount35 = typename Mma35::WarpCount;
  static int const kThreadCount35 = 32 * WarpCount35::kCount;

  using WarpCount36 = typename Mma36::WarpCount;
  static int const kThreadCount36 = 32 * WarpCount36::kCount;

  using WarpCount37 = typename Mma37::WarpCount;
  static int const kThreadCount37 = 32 * WarpCount37::kCount;

  using WarpCount38 = typename Mma38::WarpCount;
  static int const kThreadCount38 = 32 * WarpCount38::kCount;

  using WarpCount39 = typename Mma39::WarpCount;
  static int const kThreadCount39 = 32 * WarpCount39::kCount;

  using WarpCount40 = typename Mma40::WarpCount;
  static int const kThreadCount40 = 32 * WarpCount40::kCount;

  using WarpCount41 = typename Mma41::WarpCount;
  static int const kThreadCount41 = 32 * WarpCount41::kCount;

  using WarpCount42 = typename Mma42::WarpCount;
  static int const kThreadCount42 = 32 * WarpCount42::kCount;

  using WarpCount43 = typename Mma43::WarpCount;
  static int const kThreadCount43 = 32 * WarpCount43::kCount;

  using WarpCount44 = typename Mma44::WarpCount;
  static int const kThreadCount44 = 32 * WarpCount44::kCount;

  using WarpCount45 = typename Mma45::WarpCount;
  static int const kThreadCount45 = 32 * WarpCount45::kCount;

  using WarpCount46 = typename Mma46::WarpCount;
  static int const kThreadCount46 = 32 * WarpCount46::kCount;

  using WarpCount47 = typename Mma47::WarpCount;
  static int const kThreadCount47 = 32 * WarpCount47::kCount;

  using WarpCount48 = typename Mma48::WarpCount;
  static int const kThreadCount48 = 32 * WarpCount48::kCount;

  using WarpCount49 = typename Mma49::WarpCount;
  static int const kThreadCount49 = 32 * WarpCount49::kCount;

  using WarpCount50 = typename Mma50::WarpCount;
  static int const kThreadCount50 = 32 * WarpCount50::kCount;

  using WarpCount51 = typename Mma51::WarpCount;
  static int const kThreadCount51 = 32 * WarpCount51::kCount;

  using WarpCount52 = typename Mma52::WarpCount;
  static int const kThreadCount52 = 32 * WarpCount52::kCount;

  using WarpCount53 = typename Mma53::WarpCount;
  static int const kThreadCount53 = 32 * WarpCount53::kCount;


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
      typename LayoutC::Stride::LongIndex *ldd
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
      ldd(ldd)
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

    typename Mma2::SharedStorage main_loop2;
    typename Epilogue2::SharedStorage epilogue2;

    typename Mma3::SharedStorage main_loop3;
    typename Epilogue3::SharedStorage epilogue3;

    typename Mma4::SharedStorage main_loop4;
    typename Epilogue4::SharedStorage epilogue4;

    typename Mma5::SharedStorage main_loop5;
    typename Epilogue5::SharedStorage epilogue5;

    typename Mma6::SharedStorage main_loop6;
    typename Epilogue6::SharedStorage epilogue6;

    typename Mma7::SharedStorage main_loop7;
    typename Epilogue7::SharedStorage epilogue7;

    typename Mma8::SharedStorage main_loop8;
    typename Epilogue8::SharedStorage epilogue8;

    typename Mma9::SharedStorage main_loop9;
    typename Epilogue9::SharedStorage epilogue9;

    typename Mma10::SharedStorage main_loop10;
    typename Epilogue10::SharedStorage epilogue10;

    typename Mma11::SharedStorage main_loop11;
    typename Epilogue11::SharedStorage epilogue11;

    typename Mma12::SharedStorage main_loop12;
    typename Epilogue12::SharedStorage epilogue12;

    typename Mma13::SharedStorage main_loop13;
    typename Epilogue13::SharedStorage epilogue13;

    typename Mma14::SharedStorage main_loop14;
    typename Epilogue14::SharedStorage epilogue14;

    typename Mma15::SharedStorage main_loop15;
    typename Epilogue15::SharedStorage epilogue15;

    typename Mma16::SharedStorage main_loop16;
    typename Epilogue16::SharedStorage epilogue16;

    typename Mma17::SharedStorage main_loop17;
    typename Epilogue17::SharedStorage epilogue17;

    typename Mma18::SharedStorage main_loop18;
    typename Epilogue18::SharedStorage epilogue18;

    typename Mma19::SharedStorage main_loop19;
    typename Epilogue19::SharedStorage epilogue19;

    typename Mma20::SharedStorage main_loop20;
    typename Epilogue20::SharedStorage epilogue20;

    typename Mma21::SharedStorage main_loop21;
    typename Epilogue21::SharedStorage epilogue21;

    typename Mma22::SharedStorage main_loop22;
    typename Epilogue22::SharedStorage epilogue22;

    typename Mma23::SharedStorage main_loop23;
    typename Epilogue23::SharedStorage epilogue23;

    typename Mma24::SharedStorage main_loop24;
    typename Epilogue24::SharedStorage epilogue24;

    typename Mma25::SharedStorage main_loop25;
    typename Epilogue25::SharedStorage epilogue25;

    typename Mma26::SharedStorage main_loop26;
    typename Epilogue26::SharedStorage epilogue26;

    typename Mma27::SharedStorage main_loop27;
    typename Epilogue27::SharedStorage epilogue27;

    typename Mma28::SharedStorage main_loop28;
    typename Epilogue28::SharedStorage epilogue28;

    typename Mma29::SharedStorage main_loop29;
    typename Epilogue29::SharedStorage epilogue29;

    typename Mma30::SharedStorage main_loop30;
    typename Epilogue30::SharedStorage epilogue30;

    typename Mma31::SharedStorage main_loop31;
    typename Epilogue31::SharedStorage epilogue31;

    typename Mma32::SharedStorage main_loop32;
    typename Epilogue32::SharedStorage epilogue32;

    typename Mma33::SharedStorage main_loop33;
    typename Epilogue33::SharedStorage epilogue33;

    typename Mma34::SharedStorage main_loop34;
    typename Epilogue34::SharedStorage epilogue34;

    typename Mma35::SharedStorage main_loop35;
    typename Epilogue35::SharedStorage epilogue35;

    typename Mma36::SharedStorage main_loop36;
    typename Epilogue36::SharedStorage epilogue36;

    typename Mma37::SharedStorage main_loop37;
    typename Epilogue37::SharedStorage epilogue37;

    typename Mma38::SharedStorage main_loop38;
    typename Epilogue38::SharedStorage epilogue38;

    typename Mma39::SharedStorage main_loop39;
    typename Epilogue39::SharedStorage epilogue39;

    typename Mma40::SharedStorage main_loop40;
    typename Epilogue40::SharedStorage epilogue40;

    typename Mma41::SharedStorage main_loop41;
    typename Epilogue41::SharedStorage epilogue41;

    typename Mma42::SharedStorage main_loop42;
    typename Epilogue42::SharedStorage epilogue42;

    typename Mma43::SharedStorage main_loop43;
    typename Epilogue43::SharedStorage epilogue43;

    typename Mma44::SharedStorage main_loop44;
    typename Epilogue44::SharedStorage epilogue44;

    typename Mma45::SharedStorage main_loop45;
    typename Epilogue45::SharedStorage epilogue45;

    typename Mma46::SharedStorage main_loop46;
    typename Epilogue46::SharedStorage epilogue46;

    typename Mma47::SharedStorage main_loop47;
    typename Epilogue47::SharedStorage epilogue47;

    typename Mma48::SharedStorage main_loop48;
    typename Epilogue48::SharedStorage epilogue48;

    typename Mma49::SharedStorage main_loop49;
    typename Epilogue49::SharedStorage epilogue49;

    typename Mma50::SharedStorage main_loop50;
    typename Epilogue50::SharedStorage epilogue50;

    typename Mma51::SharedStorage main_loop51;
    typename Epilogue51::SharedStorage epilogue51;

    typename Mma52::SharedStorage main_loop52;
    typename Epilogue52::SharedStorage epilogue52;

    typename Mma53::SharedStorage main_loop53;
    typename Epilogue53::SharedStorage epilogue53;

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
      {Mma2::Shape::kM, Mma2::Shape::kN}, 
      {Mma3::Shape::kM, Mma3::Shape::kN}, 
      {Mma4::Shape::kM, Mma4::Shape::kN}, 
      {Mma5::Shape::kM, Mma5::Shape::kN}, 
      {Mma6::Shape::kM, Mma6::Shape::kN}, 
      {Mma7::Shape::kM, Mma7::Shape::kN}, 
      {Mma8::Shape::kM, Mma8::Shape::kN}, 
      {Mma9::Shape::kM, Mma9::Shape::kN}, 
      {Mma10::Shape::kM, Mma10::Shape::kN}, 
      {Mma11::Shape::kM, Mma11::Shape::kN}, 
      {Mma12::Shape::kM, Mma12::Shape::kN}, 
      {Mma13::Shape::kM, Mma13::Shape::kN}, 
      {Mma14::Shape::kM, Mma14::Shape::kN}, 
      {Mma15::Shape::kM, Mma15::Shape::kN}, 
      {Mma16::Shape::kM, Mma16::Shape::kN}, 
      {Mma17::Shape::kM, Mma17::Shape::kN}, 
      {Mma18::Shape::kM, Mma18::Shape::kN}, 
      {Mma19::Shape::kM, Mma19::Shape::kN}, 
      {Mma20::Shape::kM, Mma20::Shape::kN}, 
      {Mma21::Shape::kM, Mma21::Shape::kN}, 
      {Mma22::Shape::kM, Mma22::Shape::kN}, 
      {Mma23::Shape::kM, Mma23::Shape::kN}, 
      {Mma24::Shape::kM, Mma24::Shape::kN}, 
      {Mma25::Shape::kM, Mma25::Shape::kN}, 
      {Mma26::Shape::kM, Mma26::Shape::kN}, 
      {Mma27::Shape::kM, Mma27::Shape::kN}, 
      {Mma28::Shape::kM, Mma28::Shape::kN}, 
      {Mma29::Shape::kM, Mma29::Shape::kN}, 
      {Mma30::Shape::kM, Mma30::Shape::kN}, 
      {Mma31::Shape::kM, Mma31::Shape::kN}, 
      {Mma32::Shape::kM, Mma32::Shape::kN}, 
      {Mma33::Shape::kM, Mma33::Shape::kN}, 
      {Mma34::Shape::kM, Mma34::Shape::kN}, 
      {Mma35::Shape::kM, Mma35::Shape::kN}, 
      {Mma36::Shape::kM, Mma36::Shape::kN}, 
      {Mma37::Shape::kM, Mma37::Shape::kN}, 
      {Mma38::Shape::kM, Mma38::Shape::kN}, 
      {Mma39::Shape::kM, Mma39::Shape::kN}, 
      {Mma40::Shape::kM, Mma40::Shape::kN}, 
      {Mma41::Shape::kM, Mma41::Shape::kN}, 
      {Mma42::Shape::kM, Mma42::Shape::kN}, 
      {Mma43::Shape::kM, Mma43::Shape::kN}, 
      {Mma44::Shape::kM, Mma44::Shape::kN}, 
      {Mma45::Shape::kM, Mma45::Shape::kN}, 
      {Mma46::Shape::kM, Mma46::Shape::kN}, 
      {Mma47::Shape::kM, Mma47::Shape::kN}, 
      {Mma48::Shape::kM, Mma48::Shape::kN}, 
      {Mma49::Shape::kM, Mma49::Shape::kN}, 
      {Mma50::Shape::kM, Mma50::Shape::kN}, 
      {Mma51::Shape::kM, Mma51::Shape::kN}, 
      {Mma52::Shape::kM, Mma52::Shape::kN}, 
      {Mma53::Shape::kM, Mma53::Shape::kN}, 

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

        case 2:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma2::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma2::Shape::kN,
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
          typename Mma2::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma2::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma2::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma2 mma(shared_storage.main_loop2, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma2::Shape::kK - 1) / Mma2::Shape::kK;
          

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

          typename Epilogue2::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue2::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue2::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue2::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue2 epilogue(
            shared_storage.epilogue2, 
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

        case 3:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma3::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma3::Shape::kN,
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
          typename Mma3::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma3::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma3::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma3 mma(shared_storage.main_loop3, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma3::Shape::kK - 1) / Mma3::Shape::kK;
          

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

          typename Epilogue3::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue3::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue3::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue3::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue3 epilogue(
            shared_storage.epilogue3, 
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

        case 4:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma4::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma4::Shape::kN,
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
          typename Mma4::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma4::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma4::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma4 mma(shared_storage.main_loop4, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma4::Shape::kK - 1) / Mma4::Shape::kK;
          

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

          typename Epilogue4::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue4::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue4::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue4::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue4 epilogue(
            shared_storage.epilogue4, 
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

        case 5:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma5::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma5::Shape::kN,
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
          typename Mma5::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma5::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma5::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma5 mma(shared_storage.main_loop5, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma5::Shape::kK - 1) / Mma5::Shape::kK;
          

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

          typename Epilogue5::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue5::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue5::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue5::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue5 epilogue(
            shared_storage.epilogue5, 
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

        case 6:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma6::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma6::Shape::kN,
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
          typename Mma6::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma6::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma6::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma6 mma(shared_storage.main_loop6, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma6::Shape::kK - 1) / Mma6::Shape::kK;
          

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

          typename Epilogue6::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue6::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue6::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue6::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue6 epilogue(
            shared_storage.epilogue6, 
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

        case 7:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma7::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma7::Shape::kN,
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
          typename Mma7::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma7::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma7::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma7 mma(shared_storage.main_loop7, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma7::Shape::kK - 1) / Mma7::Shape::kK;
          

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

          typename Epilogue7::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue7::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue7::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue7::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue7 epilogue(
            shared_storage.epilogue7, 
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

        case 8:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma8::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma8::Shape::kN,
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
          typename Mma8::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma8::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma8::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma8 mma(shared_storage.main_loop8, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma8::Shape::kK - 1) / Mma8::Shape::kK;
          

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

          typename Epilogue8::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue8::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue8::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue8::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue8 epilogue(
            shared_storage.epilogue8, 
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

        case 9:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma9::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma9::Shape::kN,
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
          typename Mma9::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma9::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma9::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma9 mma(shared_storage.main_loop9, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma9::Shape::kK - 1) / Mma9::Shape::kK;
          

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

          typename Epilogue9::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue9::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue9::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue9::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue9 epilogue(
            shared_storage.epilogue9, 
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

        case 10:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma10::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma10::Shape::kN,
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
          typename Mma10::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma10::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma10::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma10 mma(shared_storage.main_loop10, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma10::Shape::kK - 1) / Mma10::Shape::kK;
          

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

          typename Epilogue10::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue10::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue10::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue10::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue10 epilogue(
            shared_storage.epilogue10, 
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

        case 11:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma11::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma11::Shape::kN,
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
          typename Mma11::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma11::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma11::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma11 mma(shared_storage.main_loop11, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma11::Shape::kK - 1) / Mma11::Shape::kK;
          

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

          typename Epilogue11::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue11::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue11::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue11::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue11 epilogue(
            shared_storage.epilogue11, 
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

        case 12:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma12::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma12::Shape::kN,
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
          typename Mma12::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma12::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma12::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma12 mma(shared_storage.main_loop12, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma12::Shape::kK - 1) / Mma12::Shape::kK;
          

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

          typename Epilogue12::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue12::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue12::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue12::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue12 epilogue(
            shared_storage.epilogue12, 
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

        case 13:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma13::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma13::Shape::kN,
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
          typename Mma13::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma13::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma13::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma13 mma(shared_storage.main_loop13, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma13::Shape::kK - 1) / Mma13::Shape::kK;
          

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

          typename Epilogue13::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue13::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue13::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue13::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue13 epilogue(
            shared_storage.epilogue13, 
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

        case 14:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma14::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma14::Shape::kN,
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
          typename Mma14::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma14::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma14::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma14 mma(shared_storage.main_loop14, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma14::Shape::kK - 1) / Mma14::Shape::kK;
          

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

          typename Epilogue14::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue14::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue14::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue14::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue14 epilogue(
            shared_storage.epilogue14, 
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

        case 15:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma15::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma15::Shape::kN,
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
          typename Mma15::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma15::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma15::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma15 mma(shared_storage.main_loop15, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma15::Shape::kK - 1) / Mma15::Shape::kK;
          

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

          typename Epilogue15::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue15::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue15::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue15::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue15 epilogue(
            shared_storage.epilogue15, 
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

        case 16:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma16::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma16::Shape::kN,
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
          typename Mma16::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma16::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma16::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma16 mma(shared_storage.main_loop16, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma16::Shape::kK - 1) / Mma16::Shape::kK;
          

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

          typename Epilogue16::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue16::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue16::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue16::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue16 epilogue(
            shared_storage.epilogue16, 
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

        case 17:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma17::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma17::Shape::kN,
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
          typename Mma17::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma17::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma17::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma17 mma(shared_storage.main_loop17, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma17::Shape::kK - 1) / Mma17::Shape::kK;
          

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

          typename Epilogue17::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue17::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue17::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue17::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue17 epilogue(
            shared_storage.epilogue17, 
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

        case 18:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma18::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma18::Shape::kN,
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
          typename Mma18::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma18::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma18::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma18 mma(shared_storage.main_loop18, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma18::Shape::kK - 1) / Mma18::Shape::kK;
          

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

          typename Epilogue18::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue18::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue18::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue18::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue18 epilogue(
            shared_storage.epilogue18, 
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

        case 19:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma19::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma19::Shape::kN,
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
          typename Mma19::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma19::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma19::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma19 mma(shared_storage.main_loop19, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma19::Shape::kK - 1) / Mma19::Shape::kK;
          

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

          typename Epilogue19::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue19::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue19::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue19::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue19 epilogue(
            shared_storage.epilogue19, 
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

        case 20:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma20::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma20::Shape::kN,
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
          typename Mma20::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma20::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma20::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma20 mma(shared_storage.main_loop20, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma20::Shape::kK - 1) / Mma20::Shape::kK;
          

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

          typename Epilogue20::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue20::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue20::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue20::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue20 epilogue(
            shared_storage.epilogue20, 
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

        case 21:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma21::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma21::Shape::kN,
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
          typename Mma21::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma21::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma21::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma21 mma(shared_storage.main_loop21, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma21::Shape::kK - 1) / Mma21::Shape::kK;
          

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

          typename Epilogue21::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue21::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue21::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue21::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue21 epilogue(
            shared_storage.epilogue21, 
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

        case 22:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma22::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma22::Shape::kN,
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
          typename Mma22::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma22::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma22::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma22 mma(shared_storage.main_loop22, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma22::Shape::kK - 1) / Mma22::Shape::kK;
          

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

          typename Epilogue22::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue22::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue22::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue22::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue22 epilogue(
            shared_storage.epilogue22, 
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

        case 23:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma23::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma23::Shape::kN,
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
          typename Mma23::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma23::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma23::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma23 mma(shared_storage.main_loop23, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma23::Shape::kK - 1) / Mma23::Shape::kK;
          

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

          typename Epilogue23::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue23::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue23::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue23::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue23 epilogue(
            shared_storage.epilogue23, 
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

        case 24:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma24::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma24::Shape::kN,
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
          typename Mma24::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma24::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma24::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma24 mma(shared_storage.main_loop24, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma24::Shape::kK - 1) / Mma24::Shape::kK;
          

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

          typename Epilogue24::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue24::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue24::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue24::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue24 epilogue(
            shared_storage.epilogue24, 
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

        case 25:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma25::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma25::Shape::kN,
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
          typename Mma25::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma25::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma25::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma25 mma(shared_storage.main_loop25, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma25::Shape::kK - 1) / Mma25::Shape::kK;
          

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

          typename Epilogue25::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue25::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue25::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue25::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue25 epilogue(
            shared_storage.epilogue25, 
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

        case 26:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma26::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma26::Shape::kN,
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
          typename Mma26::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma26::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma26::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma26 mma(shared_storage.main_loop26, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma26::Shape::kK - 1) / Mma26::Shape::kK;
          

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

          typename Epilogue26::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue26::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue26::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue26::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue26 epilogue(
            shared_storage.epilogue26, 
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

        case 27:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma27::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma27::Shape::kN,
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
          typename Mma27::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma27::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma27::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma27 mma(shared_storage.main_loop27, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma27::Shape::kK - 1) / Mma27::Shape::kK;
          

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

          typename Epilogue27::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue27::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue27::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue27::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue27 epilogue(
            shared_storage.epilogue27, 
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

        case 28:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma28::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma28::Shape::kN,
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
          typename Mma28::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma28::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma28::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma28 mma(shared_storage.main_loop28, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma28::Shape::kK - 1) / Mma28::Shape::kK;
          

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

          typename Epilogue28::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue28::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue28::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue28::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue28 epilogue(
            shared_storage.epilogue28, 
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

        case 29:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma29::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma29::Shape::kN,
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
          typename Mma29::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma29::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma29::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma29 mma(shared_storage.main_loop29, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma29::Shape::kK - 1) / Mma29::Shape::kK;
          

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

          typename Epilogue29::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue29::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue29::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue29::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue29 epilogue(
            shared_storage.epilogue29, 
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

        case 30:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma30::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma30::Shape::kN,
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
          typename Mma30::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma30::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma30::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma30 mma(shared_storage.main_loop30, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma30::Shape::kK - 1) / Mma30::Shape::kK;
          

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

          typename Epilogue30::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue30::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue30::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue30::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue30 epilogue(
            shared_storage.epilogue30, 
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

        case 31:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma31::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma31::Shape::kN,
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
          typename Mma31::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma31::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma31::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma31 mma(shared_storage.main_loop31, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma31::Shape::kK - 1) / Mma31::Shape::kK;
          

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

          typename Epilogue31::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue31::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue31::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue31::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue31 epilogue(
            shared_storage.epilogue31, 
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

        case 32:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma32::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma32::Shape::kN,
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
          typename Mma32::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma32::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma32::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma32 mma(shared_storage.main_loop32, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma32::Shape::kK - 1) / Mma32::Shape::kK;
          

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

          typename Epilogue32::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue32::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue32::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue32::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue32 epilogue(
            shared_storage.epilogue32, 
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

        case 33:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma33::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma33::Shape::kN,
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
          typename Mma33::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma33::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma33::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma33 mma(shared_storage.main_loop33, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma33::Shape::kK - 1) / Mma33::Shape::kK;
          

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

          typename Epilogue33::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue33::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue33::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue33::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue33 epilogue(
            shared_storage.epilogue33, 
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

        case 34:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma34::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma34::Shape::kN,
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
          typename Mma34::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma34::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma34::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma34 mma(shared_storage.main_loop34, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma34::Shape::kK - 1) / Mma34::Shape::kK;
          

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

          typename Epilogue34::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue34::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue34::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue34::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue34 epilogue(
            shared_storage.epilogue34, 
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

        case 35:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma35::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma35::Shape::kN,
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
          typename Mma35::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma35::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma35::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma35 mma(shared_storage.main_loop35, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma35::Shape::kK - 1) / Mma35::Shape::kK;
          

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

          typename Epilogue35::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue35::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue35::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue35::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue35 epilogue(
            shared_storage.epilogue35, 
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

        case 36:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma36::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma36::Shape::kN,
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
          typename Mma36::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma36::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma36::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma36 mma(shared_storage.main_loop36, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma36::Shape::kK - 1) / Mma36::Shape::kK;
          

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

          typename Epilogue36::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue36::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue36::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue36::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue36 epilogue(
            shared_storage.epilogue36, 
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

        case 37:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma37::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma37::Shape::kN,
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
          typename Mma37::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma37::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma37::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma37 mma(shared_storage.main_loop37, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma37::Shape::kK - 1) / Mma37::Shape::kK;
          

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

          typename Epilogue37::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue37::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue37::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue37::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue37 epilogue(
            shared_storage.epilogue37, 
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

        case 38:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma38::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma38::Shape::kN,
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
          typename Mma38::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma38::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma38::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma38 mma(shared_storage.main_loop38, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma38::Shape::kK - 1) / Mma38::Shape::kK;
          

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

          typename Epilogue38::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue38::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue38::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue38::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue38 epilogue(
            shared_storage.epilogue38, 
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

        case 39:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma39::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma39::Shape::kN,
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
          typename Mma39::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma39::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma39::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma39 mma(shared_storage.main_loop39, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma39::Shape::kK - 1) / Mma39::Shape::kK;
          

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

          typename Epilogue39::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue39::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue39::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue39::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue39 epilogue(
            shared_storage.epilogue39, 
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

        case 40:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma40::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma40::Shape::kN,
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
          typename Mma40::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma40::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma40::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma40 mma(shared_storage.main_loop40, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma40::Shape::kK - 1) / Mma40::Shape::kK;
          

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

          typename Epilogue40::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue40::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue40::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue40::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue40 epilogue(
            shared_storage.epilogue40, 
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

        case 41:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma41::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma41::Shape::kN,
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
          typename Mma41::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma41::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma41::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma41 mma(shared_storage.main_loop41, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma41::Shape::kK - 1) / Mma41::Shape::kK;
          

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

          typename Epilogue41::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue41::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue41::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue41::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue41 epilogue(
            shared_storage.epilogue41, 
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

        case 42:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma42::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma42::Shape::kN,
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
          typename Mma42::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma42::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma42::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma42 mma(shared_storage.main_loop42, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma42::Shape::kK - 1) / Mma42::Shape::kK;
          

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

          typename Epilogue42::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue42::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue42::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue42::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue42 epilogue(
            shared_storage.epilogue42, 
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

        case 43:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma43::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma43::Shape::kN,
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
          typename Mma43::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma43::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma43::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma43 mma(shared_storage.main_loop43, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma43::Shape::kK - 1) / Mma43::Shape::kK;
          

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

          typename Epilogue43::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue43::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue43::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue43::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue43 epilogue(
            shared_storage.epilogue43, 
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

        case 44:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma44::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma44::Shape::kN,
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
          typename Mma44::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma44::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma44::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma44 mma(shared_storage.main_loop44, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma44::Shape::kK - 1) / Mma44::Shape::kK;
          

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

          typename Epilogue44::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue44::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue44::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue44::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue44 epilogue(
            shared_storage.epilogue44, 
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

        case 45:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma45::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma45::Shape::kN,
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
          typename Mma45::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma45::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma45::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma45 mma(shared_storage.main_loop45, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma45::Shape::kK - 1) / Mma45::Shape::kK;
          

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

          typename Epilogue45::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue45::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue45::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue45::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue45 epilogue(
            shared_storage.epilogue45, 
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

        case 46:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma46::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma46::Shape::kN,
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
          typename Mma46::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma46::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma46::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma46 mma(shared_storage.main_loop46, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma46::Shape::kK - 1) / Mma46::Shape::kK;
          

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

          typename Epilogue46::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue46::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue46::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue46::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue46 epilogue(
            shared_storage.epilogue46, 
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

        case 47:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma47::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma47::Shape::kN,
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
          typename Mma47::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma47::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma47::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma47 mma(shared_storage.main_loop47, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma47::Shape::kK - 1) / Mma47::Shape::kK;
          

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

          typename Epilogue47::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue47::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue47::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue47::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue47 epilogue(
            shared_storage.epilogue47, 
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

        case 48:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma48::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma48::Shape::kN,
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
          typename Mma48::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma48::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma48::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma48 mma(shared_storage.main_loop48, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma48::Shape::kK - 1) / Mma48::Shape::kK;
          

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

          typename Epilogue48::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue48::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue48::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue48::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue48 epilogue(
            shared_storage.epilogue48, 
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

        case 49:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma49::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma49::Shape::kN,
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
          typename Mma49::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma49::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma49::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma49 mma(shared_storage.main_loop49, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma49::Shape::kK - 1) / Mma49::Shape::kK;
          

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

          typename Epilogue49::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue49::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue49::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue49::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue49 epilogue(
            shared_storage.epilogue49, 
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

        case 50:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma50::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma50::Shape::kN,
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
          typename Mma50::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma50::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma50::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma50 mma(shared_storage.main_loop50, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma50::Shape::kK - 1) / Mma50::Shape::kK;
          

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

          typename Epilogue50::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue50::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue50::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue50::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue50 epilogue(
            shared_storage.epilogue50, 
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

        case 51:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma51::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma51::Shape::kN,
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
          typename Mma51::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma51::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma51::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma51 mma(shared_storage.main_loop51, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma51::Shape::kK - 1) / Mma51::Shape::kK;
          

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

          typename Epilogue51::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue51::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue51::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue51::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue51 epilogue(
            shared_storage.epilogue51, 
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

        case 52:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma52::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma52::Shape::kN,
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
          typename Mma52::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma52::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma52::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma52 mma(shared_storage.main_loop52, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma52::Shape::kK - 1) / Mma52::Shape::kK;
          

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

          typename Epilogue52::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue52::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue52::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue52::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue52 epilogue(
            shared_storage.epilogue52, 
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

        case 53:
        {
          GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

          cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma53::Shape::kM,
            int(cta_idx % grid_shape.n()) * Mma53::Shape::kN,
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
          typename Mma53::IteratorA iterator_A(
            LayoutA(ldm_A),
            ptr_A,
            {problem_size.m(), problem_size.k()},
            thread_idx,
            tb_offset_A);

          typename Mma53::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k(), problem_size.n()},
            thread_idx,
            tb_offset_B);

          typename Mma53::FragmentC accumulators;

          accumulators.clear();
          
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

          int lane_idx = threadIdx.x % 32;

          //
          // Matrix multiply phase
          //

          // Construct thread-scoped matrix multiply
          Mma53 mma(shared_storage.main_loop53, thread_idx, warp_idx, lane_idx);

          // Compute threadblock-scoped matrix multiply-add
          int gemm_k_iterations = (problem_size.k() + Mma53::Shape::kK - 1) / Mma53::Shape::kK;
          

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

          typename Epilogue53::OutputTileIterator::Params params_C(layout_C);
          typename Epilogue53::OutputTileIterator::Params params_D(layout_D);

          // Tile iterator loading from source tensor.
          typename Epilogue53::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          // Tile iterator writing to destination tensor.
          typename Epilogue53::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn()
          );

          Epilogue53 epilogue(
            shared_storage.epilogue53, 
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

