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

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_flex_grouped_54.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
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
    typename ThreadblockShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape0,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape1,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape2,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape2,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape2,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape3,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape3,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape3,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape4,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape4,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape4,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape5,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape5,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape5,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape6,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape6,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape6,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape7,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape7,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape7,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape8,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape8,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape8,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape9,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape9,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape9,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape10,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape10,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape10,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape11,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape11,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape11,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape12,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape12,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape12,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape13,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape13,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape13,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape14,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape14,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape14,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape15,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape15,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape15,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape16,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape16,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape16,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape17,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape17,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape17,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape18,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape18,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape18,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape19,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape19,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape19,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape20,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape20,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape20,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape21,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape21,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape21,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape22,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape22,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape22,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape23,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape23,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape23,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape24,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape24,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape24,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape25,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape25,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape25,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape26,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape26,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape26,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape27,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape27,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape27,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape28,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape28,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape28,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape29,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape29,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape29,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape30,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape30,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape30,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape31,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape31,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape31,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape32,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape32,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape32,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape33,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape33,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape33,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape34,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape34,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape34,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape35,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape35,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape35,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape36,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape36,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape36,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape37,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape37,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape37,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape38,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape38,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape38,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape39,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape39,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape39,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape40,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape40,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape40,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape41,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape41,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape41,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape42,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape42,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape42,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape43,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape43,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape43,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape44,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape44,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape44,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape45,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape45,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape45,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape46,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape46,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape46,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape47,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape47,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape47,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape48,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape48,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape48,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape49,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape49,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape49,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape50,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape50,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape50,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape51,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape51,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape51,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape52,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape52,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape52,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape53,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape53,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape53,

    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator = typename device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA_, ElementB_, ElementC_,
        ElementAccumulator>::Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    ///
    typename Enable = void
    >
struct DefaultGemmFlexGrouped;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued GEMM kernels
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
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
    typename ThreadblockShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape0,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape1,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape2,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape2,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape2,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape3,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape3,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape3,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape4,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape4,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape4,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape5,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape5,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape5,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape6,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape6,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape6,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape7,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape7,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape7,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape8,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape8,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape8,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape9,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape9,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape9,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape10,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape10,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape10,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape11,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape11,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape11,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape12,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape12,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape12,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape13,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape13,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape13,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape14,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape14,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape14,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape15,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape15,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape15,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape16,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape16,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape16,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape17,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape17,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape17,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape18,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape18,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape18,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape19,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape19,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape19,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape20,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape20,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape20,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape21,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape21,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape21,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape22,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape22,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape22,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape23,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape23,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape23,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape24,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape24,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape24,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape25,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape25,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape25,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape26,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape26,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape26,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape27,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape27,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape27,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape28,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape28,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape28,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape29,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape29,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape29,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape30,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape30,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape30,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape31,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape31,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape31,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape32,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape32,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape32,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape33,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape33,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape33,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape34,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape34,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape34,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape35,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape35,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape35,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape36,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape36,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape36,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape37,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape37,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape37,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape38,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape38,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape38,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape39,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape39,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape39,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape40,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape40,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape40,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape41,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape41,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape41,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape42,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape42,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape42,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape43,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape43,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape43,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape44,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape44,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape44,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape45,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape45,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape45,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape46,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape46,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape46,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape47,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape47,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape47,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape48,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape48,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape48,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape49,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape49,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape49,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape50,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape50,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape50,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape51,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape51,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape51,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape52,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape52,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape52,

    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape53,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape53,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape53,

    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear
>
struct DefaultGemmFlexGrouped<
  ElementA,
  LayoutA,
  ComplexTransform::kNone,   // transform A
  kAlignmentA,
  ElementB,
  LayoutB,
  ComplexTransform::kNone,   // transform B
  kAlignmentB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  
  ThreadblockShape0,
  WarpShape0,
  InstructionShape0,

  ThreadblockShape1,
  WarpShape1,
  InstructionShape1,

  ThreadblockShape2,
  WarpShape2,
  InstructionShape2,

  ThreadblockShape3,
  WarpShape3,
  InstructionShape3,

  ThreadblockShape4,
  WarpShape4,
  InstructionShape4,

  ThreadblockShape5,
  WarpShape5,
  InstructionShape5,

  ThreadblockShape6,
  WarpShape6,
  InstructionShape6,

  ThreadblockShape7,
  WarpShape7,
  InstructionShape7,

  ThreadblockShape8,
  WarpShape8,
  InstructionShape8,

  ThreadblockShape9,
  WarpShape9,
  InstructionShape9,

  ThreadblockShape10,
  WarpShape10,
  InstructionShape10,

  ThreadblockShape11,
  WarpShape11,
  InstructionShape11,

  ThreadblockShape12,
  WarpShape12,
  InstructionShape12,

  ThreadblockShape13,
  WarpShape13,
  InstructionShape13,

  ThreadblockShape14,
  WarpShape14,
  InstructionShape14,

  ThreadblockShape15,
  WarpShape15,
  InstructionShape15,

  ThreadblockShape16,
  WarpShape16,
  InstructionShape16,

  ThreadblockShape17,
  WarpShape17,
  InstructionShape17,

  ThreadblockShape18,
  WarpShape18,
  InstructionShape18,

  ThreadblockShape19,
  WarpShape19,
  InstructionShape19,

  ThreadblockShape20,
  WarpShape20,
  InstructionShape20,

  ThreadblockShape21,
  WarpShape21,
  InstructionShape21,

  ThreadblockShape22,
  WarpShape22,
  InstructionShape22,

  ThreadblockShape23,
  WarpShape23,
  InstructionShape23,

  ThreadblockShape24,
  WarpShape24,
  InstructionShape24,

  ThreadblockShape25,
  WarpShape25,
  InstructionShape25,

  ThreadblockShape26,
  WarpShape26,
  InstructionShape26,

  ThreadblockShape27,
  WarpShape27,
  InstructionShape27,

  ThreadblockShape28,
  WarpShape28,
  InstructionShape28,

  ThreadblockShape29,
  WarpShape29,
  InstructionShape29,

  ThreadblockShape30,
  WarpShape30,
  InstructionShape30,

  ThreadblockShape31,
  WarpShape31,
  InstructionShape31,

  ThreadblockShape32,
  WarpShape32,
  InstructionShape32,

  ThreadblockShape33,
  WarpShape33,
  InstructionShape33,

  ThreadblockShape34,
  WarpShape34,
  InstructionShape34,

  ThreadblockShape35,
  WarpShape35,
  InstructionShape35,

  ThreadblockShape36,
  WarpShape36,
  InstructionShape36,

  ThreadblockShape37,
  WarpShape37,
  InstructionShape37,

  ThreadblockShape38,
  WarpShape38,
  InstructionShape38,

  ThreadblockShape39,
  WarpShape39,
  InstructionShape39,

  ThreadblockShape40,
  WarpShape40,
  InstructionShape40,

  ThreadblockShape41,
  WarpShape41,
  InstructionShape41,

  ThreadblockShape42,
  WarpShape42,
  InstructionShape42,

  ThreadblockShape43,
  WarpShape43,
  InstructionShape43,

  ThreadblockShape44,
  WarpShape44,
  InstructionShape44,

  ThreadblockShape45,
  WarpShape45,
  InstructionShape45,

  ThreadblockShape46,
  WarpShape46,
  InstructionShape46,

  ThreadblockShape47,
  WarpShape47,
  InstructionShape47,

  ThreadblockShape48,
  WarpShape48,
  InstructionShape48,

  ThreadblockShape49,
  WarpShape49,
  InstructionShape49,

  ThreadblockShape50,
  WarpShape50,
  InstructionShape50,

  ThreadblockShape51,
  WarpShape51,
  InstructionShape51,

  ThreadblockShape52,
  WarpShape52,
  InstructionShape52,

  ThreadblockShape53,
  WarpShape53,
  InstructionShape53,

  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  Operator,
  SharedMemoryClear,
  typename std::enable_if< ! cutlass::is_complex<ElementAccumulator>::value>::type
> {

  // If true, we must construct a 'transposed-and-exchanged' Mma operator.
  static bool const kInternalTranspose = platform::is_same<LayoutC, layout::ColumnMajor>::value;

  using MapArguments = kernel::detail::MapArguments<
    ElementA,
    LayoutA,
    ComplexTransform::kNone,
    kAlignmentA,
    ElementB,
    LayoutB,
    ComplexTransform::kNone,
    kAlignmentB,
    LayoutC,
    kInternalTranspose
  >;

  // Define the default GEMM kernel
  using DefaultGemmKernel0 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape0,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel1 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape1,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel2 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape2,
    WarpShape2,
    InstructionShape2,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel3 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape3,
    WarpShape3,
    InstructionShape3,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel4 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape4,
    WarpShape4,
    InstructionShape4,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel5 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape5,
    WarpShape5,
    InstructionShape5,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel6 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape6,
    WarpShape6,
    InstructionShape6,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel7 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape7,
    WarpShape7,
    InstructionShape7,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel8 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape8,
    WarpShape8,
    InstructionShape8,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel9 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape9,
    WarpShape9,
    InstructionShape9,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel10 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape10,
    WarpShape10,
    InstructionShape10,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel11 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape11,
    WarpShape11,
    InstructionShape11,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel12 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape12,
    WarpShape12,
    InstructionShape12,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel13 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape13,
    WarpShape13,
    InstructionShape13,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel14 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape14,
    WarpShape14,
    InstructionShape14,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel15 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape15,
    WarpShape15,
    InstructionShape15,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel16 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape16,
    WarpShape16,
    InstructionShape16,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel17 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape17,
    WarpShape17,
    InstructionShape17,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel18 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape18,
    WarpShape18,
    InstructionShape18,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel19 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape19,
    WarpShape19,
    InstructionShape19,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel20 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape20,
    WarpShape20,
    InstructionShape20,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel21 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape21,
    WarpShape21,
    InstructionShape21,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel22 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape22,
    WarpShape22,
    InstructionShape22,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel23 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape23,
    WarpShape23,
    InstructionShape23,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel24 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape24,
    WarpShape24,
    InstructionShape24,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel25 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape25,
    WarpShape25,
    InstructionShape25,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel26 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape26,
    WarpShape26,
    InstructionShape26,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel27 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape27,
    WarpShape27,
    InstructionShape27,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel28 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape28,
    WarpShape28,
    InstructionShape28,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel29 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape29,
    WarpShape29,
    InstructionShape29,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel30 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape30,
    WarpShape30,
    InstructionShape30,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel31 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape31,
    WarpShape31,
    InstructionShape31,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel32 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape32,
    WarpShape32,
    InstructionShape32,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel33 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape33,
    WarpShape33,
    InstructionShape33,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel34 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape34,
    WarpShape34,
    InstructionShape34,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel35 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape35,
    WarpShape35,
    InstructionShape35,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel36 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape36,
    WarpShape36,
    InstructionShape36,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel37 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape37,
    WarpShape37,
    InstructionShape37,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel38 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape38,
    WarpShape38,
    InstructionShape38,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel39 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape39,
    WarpShape39,
    InstructionShape39,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel40 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape40,
    WarpShape40,
    InstructionShape40,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel41 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape41,
    WarpShape41,
    InstructionShape41,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel42 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape42,
    WarpShape42,
    InstructionShape42,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel43 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape43,
    WarpShape43,
    InstructionShape43,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel44 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape44,
    WarpShape44,
    InstructionShape44,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel45 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape45,
    WarpShape45,
    InstructionShape45,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel46 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape46,
    WarpShape46,
    InstructionShape46,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel47 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape47,
    WarpShape47,
    InstructionShape47,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel48 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape48,
    WarpShape48,
    InstructionShape48,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel49 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape49,
    WarpShape49,
    InstructionShape49,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel50 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape50,
    WarpShape50,
    InstructionShape50,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel51 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape51,
    WarpShape51,
    InstructionShape51,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel52 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape52,
    WarpShape52,
    InstructionShape52,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

  // Define the default GEMM kernel
  using DefaultGemmKernel53 = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape53,
    WarpShape53,
    InstructionShape53,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear
  >::GemmKernel;

    /// Define the kernel in terms of the default kernel
  using GemmKernel = kernel::GemmFlexGrouped<
typename DefaultGemmKernel0::Mma,
typename DefaultGemmKernel1::Mma,
typename DefaultGemmKernel2::Mma,
typename DefaultGemmKernel3::Mma,
typename DefaultGemmKernel4::Mma,
typename DefaultGemmKernel5::Mma,
typename DefaultGemmKernel6::Mma,
typename DefaultGemmKernel7::Mma,
typename DefaultGemmKernel8::Mma,
typename DefaultGemmKernel9::Mma,
typename DefaultGemmKernel10::Mma,
typename DefaultGemmKernel11::Mma,
typename DefaultGemmKernel12::Mma,
typename DefaultGemmKernel13::Mma,
typename DefaultGemmKernel14::Mma,
typename DefaultGemmKernel15::Mma,
typename DefaultGemmKernel16::Mma,
typename DefaultGemmKernel17::Mma,
typename DefaultGemmKernel18::Mma,
typename DefaultGemmKernel19::Mma,
typename DefaultGemmKernel20::Mma,
typename DefaultGemmKernel21::Mma,
typename DefaultGemmKernel22::Mma,
typename DefaultGemmKernel23::Mma,
typename DefaultGemmKernel24::Mma,
typename DefaultGemmKernel25::Mma,
typename DefaultGemmKernel26::Mma,
typename DefaultGemmKernel27::Mma,
typename DefaultGemmKernel28::Mma,
typename DefaultGemmKernel29::Mma,
typename DefaultGemmKernel30::Mma,
typename DefaultGemmKernel31::Mma,
typename DefaultGemmKernel32::Mma,
typename DefaultGemmKernel33::Mma,
typename DefaultGemmKernel34::Mma,
typename DefaultGemmKernel35::Mma,
typename DefaultGemmKernel36::Mma,
typename DefaultGemmKernel37::Mma,
typename DefaultGemmKernel38::Mma,
typename DefaultGemmKernel39::Mma,
typename DefaultGemmKernel40::Mma,
typename DefaultGemmKernel41::Mma,
typename DefaultGemmKernel42::Mma,
typename DefaultGemmKernel43::Mma,
typename DefaultGemmKernel44::Mma,
typename DefaultGemmKernel45::Mma,
typename DefaultGemmKernel46::Mma,
typename DefaultGemmKernel47::Mma,
typename DefaultGemmKernel48::Mma,
typename DefaultGemmKernel49::Mma,
typename DefaultGemmKernel50::Mma,
typename DefaultGemmKernel51::Mma,
typename DefaultGemmKernel52::Mma,
typename DefaultGemmKernel53::Mma,
typename DefaultGemmKernel0::Epilogue,
typename DefaultGemmKernel1::Epilogue,
typename DefaultGemmKernel2::Epilogue,
typename DefaultGemmKernel3::Epilogue,
typename DefaultGemmKernel4::Epilogue,
typename DefaultGemmKernel5::Epilogue,
typename DefaultGemmKernel6::Epilogue,
typename DefaultGemmKernel7::Epilogue,
typename DefaultGemmKernel8::Epilogue,
typename DefaultGemmKernel9::Epilogue,
typename DefaultGemmKernel10::Epilogue,
typename DefaultGemmKernel11::Epilogue,
typename DefaultGemmKernel12::Epilogue,
typename DefaultGemmKernel13::Epilogue,
typename DefaultGemmKernel14::Epilogue,
typename DefaultGemmKernel15::Epilogue,
typename DefaultGemmKernel16::Epilogue,
typename DefaultGemmKernel17::Epilogue,
typename DefaultGemmKernel18::Epilogue,
typename DefaultGemmKernel19::Epilogue,
typename DefaultGemmKernel20::Epilogue,
typename DefaultGemmKernel21::Epilogue,
typename DefaultGemmKernel22::Epilogue,
typename DefaultGemmKernel23::Epilogue,
typename DefaultGemmKernel24::Epilogue,
typename DefaultGemmKernel25::Epilogue,
typename DefaultGemmKernel26::Epilogue,
typename DefaultGemmKernel27::Epilogue,
typename DefaultGemmKernel28::Epilogue,
typename DefaultGemmKernel29::Epilogue,
typename DefaultGemmKernel30::Epilogue,
typename DefaultGemmKernel31::Epilogue,
typename DefaultGemmKernel32::Epilogue,
typename DefaultGemmKernel33::Epilogue,
typename DefaultGemmKernel34::Epilogue,
typename DefaultGemmKernel35::Epilogue,
typename DefaultGemmKernel36::Epilogue,
typename DefaultGemmKernel37::Epilogue,
typename DefaultGemmKernel38::Epilogue,
typename DefaultGemmKernel39::Epilogue,
typename DefaultGemmKernel40::Epilogue,
typename DefaultGemmKernel41::Epilogue,
typename DefaultGemmKernel42::Epilogue,
typename DefaultGemmKernel43::Epilogue,
typename DefaultGemmKernel44::Epilogue,
typename DefaultGemmKernel45::Epilogue,
typename DefaultGemmKernel46::Epilogue,
typename DefaultGemmKernel47::Epilogue,
typename DefaultGemmKernel48::Epilogue,
typename DefaultGemmKernel49::Epilogue,
typename DefaultGemmKernel50::Epilogue,
typename DefaultGemmKernel51::Epilogue,
typename DefaultGemmKernel52::Epilogue,
typename DefaultGemmKernel53::Epilogue,

    ThreadblockSwizzle,
    kInternalTranspose
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
