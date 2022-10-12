

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
    \brief GEMM Grouped Example.

    This workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
    in Global Memory are passed to the kernel in array (also held in Global Memory). Similarly,
    leading dimensions and problem sizes are stored in arrays in GMEM.

    This differs from "Batched Array" GEMM because the size of each GEMM problem in the Grouped GEMM
    concept may be distinct. 

    This benchmark program initializes a workspace with random problem sizes for a given number of
    groups. Command line options enable overriding M, N, and/or K dimensions with uniform values to
    model problems more similar to the traditional batched GEMM.

    Additionally, problem sizes are collected and binned to compute the same problem as a series of
    conventional batched GEMMs (setup for this problem is not timed). This demonstrates the performance
    enhancement achieved by implementing a specialized grouped GEMM kernel.

    Examples:

    # Runs a grouped GEMM with 100 random problem sizes
    $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100

    # Runs a grouped GEMM with 100 random problem sizes (with GEMM-K dimension equal to 1024)
    $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --k=1024 --verbose=true

    # Runs a grouped GEMM that is equivalent to a batched GEMM
    $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --m=2048 --n=1024 --k=1024 --verbose=true

    # Execute Grouped GEMM and profile with NSight
    $ nv-nsight-cu-cli ./examples/24_gemm_grouped/24_gemm_grouped --m=256 --n=256 --k=256 --verbose=true \
                                                                    --iterations=1 --reference-check=false

*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <type_traits>
#include <typeinfo>
#include <string>
#include <cxxabi.h>
#include <iomanip>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution_grouped.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad_grouped.h"
#include "cutlass/conv/device/implicit_gemm_convolution_grouped.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

double runtime_ms;
double gflops;
cutlass::Status status;
cudaError_t error;
bool passed;

//
// Methods
//

Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

bool help;
bool error;
bool reference_check;

std::vector<cutlass::conv::Conv2dProblemSize> problem_sizes;

int alignment;
int problem_count;
int iterations;
int cuda_streams;
bool verbose;
float alpha;
float beta;
std::string benchmark_path;

std::string   output_tag;
std::ofstream output_file;

//
// Methods
// 

Options():
    help(false),
    error(false),
    alignment(8),
    reference_check(true),
    problem_count(15),
    iterations(20),
    cuda_streams(0),
    verbose(false),
    alpha(1),
    beta()
{ }

// Parses the command line
void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
        help = true;
        return;
    }

    cmd.get_cmd_line_argument("alignment", alignment, 8);
    cmd.get_cmd_line_argument("groups", problem_count, 15);
    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);    
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("streams", cuda_streams, 0);
    cmd.get_cmd_line_argument("verbose", verbose, false);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);

    std::string output_path;
    cmd.get_cmd_line_argument("tag", output_tag);
    cmd.get_cmd_line_argument("output_file", output_path);

    if (!output_path.empty()) {

        std::ios_base::openmode open_mode = std::ios_base::out;

        std::ifstream input_file(output_path.c_str());

        if (input_file.good()) {
            open_mode = std::ios_base::app;
            input_file.close();
        }

        output_file.open(output_path.c_str(), open_mode);

        if (output_file.good() && open_mode != std::ios_base::app) {
            output_file << "Tag,Provider,Kind,Groups,Runtime,GFLOPs\n";
        }
    }

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
        if (!benchmark_problems()) {
            error = true;
            problem_sizes.clear();
            return;
        }
    }
    else {
        error = true;
    }
}

/// Load a benchmark
bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
        return false;
    }

    while (file.good()) {

        int idx = -1;
        std::string extent_str;

        file >> idx >> extent_str;

        if (idx < 0 || extent_str.empty()) {
            break;
        }

        std::vector<int> extent;
        extent.resize(15);
        std::vector<std::string> tokens;

        cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

        for (int i = 0; i < int(tokens.size()); ++i) {
            int x = std::atoi(tokens.at(i).c_str());
            extent.at(i) = x;
        }
        cutlass::conv::Conv2dProblemSize problem(
            extent[0], extent[1], extent[2], extent[3], extent[4],
            extent[5], extent[6], extent[7], extent[8], extent[9],
            extent[10], extent[11], extent[12], extent[13], extent[14],
            cutlass::conv::Mode::kCrossCorrelation, (int)(extent[7] * extent[8] / 1024) + 1);

        problem_sizes.push_back(problem);
    }

    return true;
}

/// Prints the usage statement.
std::ostream & print_usage(std::ostream &out) const {

    out << "24_gemm_grouped\n\n"
    << "  This example profiles the performance of a 'grouped' GEMM kernel. This is similar to batched GEMM\n"
    << "  in that multiple, independent GEMMs are computed by one grid launch. It differs in that each\n"
    << "  'group' may compute a unique problem size. Problem sizes and pointers to matrices are both stored\n"
    << "  in device Global Memory and loaded by the kernel.\n\n"
    << "Options:\n\n"
    << "  --help                      If specified, displays this usage statement.\n\n"
    << "  --benchmark=<str>           Executes a benchmark problem size.\n"
    << "  --output_file=<str>         Path to a CSV file to output results. If it exists already, results are appended.\n"
    << "  --tag=<str>                 String tag to prepend to the CSV file.\n"
    << "  --groups=<int>              Number of individual GEMM problems (default: --groups=15)\n"
    << "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
    << "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
    << "  --iterations=<int>          Number of profiling iterations to perform.\n"
    << "  --reference-check=<bool>    If true, performs reference check.\n"
    << "  --verbose=<bool>            If true, prints problem sizes and batching structure.\n";

    out << "\n\nExamples:\n\n"

    << "# Runs a grouped GEMM with 100 random problem sizes\n"
    << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100\n\n"

    << "# Runs a grouped GEMM with 100 random problem sizes (with GEMM-K dimension equal to 1024)\n"
    << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --k=1024 --verbose=true\n\n"

    << "# Runs a grouped GEMM that is equivalent to a batched GEMM\n"
    << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --m=2048 --n=1024 --k=1024 --verbose=true\n\n"

    << "# Runs a grouped GEMM problem given an externally supplied benchmark file. This is a text file in which\n"
    << "# Each line contains a unique group index and an MxNxK triple indicating problemsize.\n"
    << "#\n"
    << "# For example, assume the following are the contents of 'problems.txt'\n"
    << "#\n"
    << "# 0 1024x256x520\n"
    << "# 1 520x264x1024\n"
    << "# 2 96x48x1024\n"
    << "#\n"
    << "$ ./examples/24_gemm_grouped/24_gemm_grouped --benchmark=problems.txt\n\n"

    << "# Execute Grouped GEMM and profile with NSight\n"
    << "$ nv-nsight-cu-cli ./examples/24_gemm_grouped/24_gemm_grouped --m=256 --n=256 --k=256 --verbose=true --iterations=1 --reference-check=false\n\n";

    return out;
}

/// Compute performance in GFLOP/s
double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = int64_t();

    for (auto const & problem : problem_sizes) {
        fmas += implicit_gemm_problem_size(
            cutlass::conv::Operator::kWgrad, problem).product();
    }
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
}
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ImplicitGemmConvolution>
class TestbedGrouped {
public:

//
// Type definitions
//

using ElementA = typename ImplicitGemmConvolution::ElementA;
using ElementB = typename ImplicitGemmConvolution::ElementB;
using ElementC = typename ImplicitGemmConvolution::EpilogueOutputOp::ElementOutput;
using ElementAccumulator = typename ImplicitGemmConvolution::ElementAccumulator;

using EpilogueOutputOp = typename ImplicitGemmConvolution::ImplicitGemmConvolutionkernel::Epilogue::OutputOp;
using ElementCompute = typename EpilogueOutputOp::ElementCompute;

using LayoutA = typename ImplicitGemmConvolution::LayoutA;
using LayoutB = typename ImplicitGemmConvolution::LayoutB;
using LayoutC = typename ImplicitGemmConvolution::LayoutC;

using TensorRefA = typename ImplicitGemmConvolution::TensorRefA;
using TensorRefB = typename ImplicitGemmConvolution::TensorRefB;
using TensorRefC = typename ImplicitGemmConvolution::TensorRefC;
using TensorRefD = typename ImplicitGemmConvolution::TensorRefD;

using MatrixCoord = typename LayoutC::TensorCoord;

private:

//
// Data members
//

Options & options;

/// Initialization
cutlass::Distribution::Kind init_A;
cutlass::Distribution::Kind init_B;
cutlass::Distribution::Kind init_C;
uint32_t seed;

cutlass::DeviceAllocation<cutlass::conv::Conv2dProblemSize> problem_sizes_device;

std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<ElementC> block_D;

std::vector<cutlass::HostTensor<ElementA, LayoutA>> tensor_A_host;
std::vector<cutlass::HostTensor<ElementB, LayoutB>> tensor_B_host;
std::vector<cutlass::HostTensor<ElementC, LayoutC>> tensor_C_host;
std::vector<cutlass::HostTensor<ElementC, LayoutC>> tensor_D_host;

std::vector<TensorRefA> ref_A_host;
std::vector<TensorRefB> ref_B_host;
std::vector<TensorRefC> ref_C_host;
std::vector<TensorRefD> ref_D_host;

cutlass::DeviceAllocation<TensorRefA> ref_A;
cutlass::DeviceAllocation<TensorRefB> ref_B;
cutlass::DeviceAllocation<TensorRefC> ref_C;
cutlass::DeviceAllocation<TensorRefD> ref_D;

public:

//
// Methods
//

TestbedGrouped(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
):
    options(options_), init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

int problem_count() const {
    return options.problem_sizes.size();
}

private:

/// Verbose printing of problem sizes
void print_problem_sizes_() {

    // Print groups
    std::cout << problem_count() << " groups:\n";  
    
    int32_t idx = 0;
    int64_t total_tiles = 0;

    for (auto const & problem : options.problem_sizes) {

        cutlass::gemm::GemmCoord gemm_extent(implicit_gemm_problem_size(cutlass::conv::Operator::kWgrad, problem));

        // cutlass::gemm::GemmCoord grid = this->grid_shape(cutlass::conv::Operator::kWgrad, problem);

        // int tiles = 
        //     ((problem.m() + Gemm::ThreadblockShape::kM - 1) / Gemm::ThreadblockShape::kM) * 
        //     ((problem.n() + Gemm::ThreadblockShape::kN - 1) / Gemm::ThreadblockShape::kN);

        int tiles = 0;
        total_tiles += tiles;

        std::cout << "  [" << idx << "]: " 
            << gemm_extent.m() << "-by-" << gemm_extent.n() << "-by-" << gemm_extent.k() 
            << " (" << tiles << " threadblock tiles)" << "\n";

        ++idx;
    }
}

/// Initializes data structures
void initialize_() {

    //
    // Choose random problem sizes
    //

    // construct a few problems of random sizes
    srand(seed);

    // int64_t total_elements_A = 0;
    // int64_t total_elements_B = 0;
    // int64_t total_elements_C = 0;
    // int64_t total_elements_D = 0;

    // for (int32_t i = 0; i < problem_count(); ++i) {
    //     cutlass::Tensor4DCoord extent_A = problem.output_extent();
    //     cutlass::Tensor4DCoord extent_B = problem.activation_extent();
    //     cutlass::Tensor4DCoord extent_C = problem.filter_extent();
    //     cutlass::Tensor4DCoord extent_D = problem.filter_extent();

    //     total_elements_A += extent_A.product();
    //     total_elements_B += extent_B.product();
    //     total_elements_C += extent_C.product();
    //     total_elements_D += extent_D.product();
    // }

    ref_A_host.resize(problem_count());
    ref_B_host.resize(problem_count());
    ref_C_host.resize(problem_count());
    ref_D_host.resize(problem_count());

    tensor_A_host.resize(problem_count());
    tensor_B_host.resize(problem_count());
    tensor_C_host.resize(problem_count());
    tensor_D_host.resize(problem_count());

    for (int32_t i = 0; i < problem_count(); ++i) {

        auto problem = options.problem_sizes.at(i);

        tensor_A_host.at(i) = cutlass::HostTensor<ElementA, LayoutA>(problem.output_extent());
        tensor_B_host.at(i) = cutlass::HostTensor<ElementB, LayoutB>(problem.activation_extent());            
        tensor_C_host.at(i) = cutlass::HostTensor<ElementC, LayoutC>(problem.filter_extent());            
        tensor_D_host.at(i) = cutlass::HostTensor<ElementC, LayoutC>(problem.filter_extent());

        cutlass::reference::host::TensorFillRandomUniform(
            tensor_A_host.at(i).host_view(),
            1,
            ElementA(7),
            ElementA(-8),
            0);

        cutlass::reference::host::TensorFillRandomUniform(
            tensor_B_host.at(i).host_view(),
            1,
            ElementB(7),
            ElementB(-8),
            0);

        cutlass::reference::host::TensorFill(tensor_C_host.at(i).host_view());
        cutlass::reference::host::TensorFill(tensor_D_host.at(i).host_view());

        tensor_A_host.at(i).sync_device();
        tensor_B_host.at(i).sync_device();
        tensor_C_host.at(i).sync_device();
        tensor_D_host.at(i).sync_device();
        
        ref_A_host.at(i) = tensor_A_host.at(i).device_ref();
        ref_B_host.at(i) = tensor_B_host.at(i).device_ref();
        ref_C_host.at(i) = tensor_C_host.at(i).device_ref();
        ref_D_host.at(i) = tensor_D_host.at(i).device_ref();

        std::cout << ref_A_host.at(i).data() << std::endl;
    }

    problem_sizes_device.reset(problem_count());
    problem_sizes_device.copy_from_host(options.problem_sizes.data());

    ref_A.reset(problem_count());
    ref_B.reset(problem_count());
    ref_C.reset(problem_count());
    ref_D.reset(problem_count());

    ref_A.copy_from_host(ref_A_host.data());
    ref_B.copy_from_host(ref_B_host.data());
    ref_C.copy_from_host(ref_C_host.data());
    ref_D.copy_from_host(ref_D_host.data());

    cudaDeviceSynchronize();
}

/// Verifies the result is a GEMM
bool verify_() {

    bool passed = true;

    for (int32_t i = 0; i < problem_count(); ++i) {
        cutlass::conv::Conv2dProblemSize problem = options.problem_sizes.at(i);

        cutlass::HostTensor<ElementC, LayoutC> tensor_d(problem.filter_extent());

        cutlass::reference::host::TensorFill(tensor_d.host_view());
        tensor_d.sync_device();

        // Reference GEMM
        cutlass::reference::device::Conv2dWgrad<
            ElementA, LayoutA,
            ElementB, LayoutB,
            ElementC, LayoutC, 
            ElementCompute, ElementAccumulator
        >(
            problem,
            tensor_A_host.at(i).device_ref(),
            tensor_B_host.at(i).device_ref(),
            tensor_C_host.at(i).device_ref(),
            tensor_d.device_ref(),
            options.alpha,
            options.beta
        );

        // Copy to host memory
        tensor_C_host.at(i).sync_host();
        tensor_d.sync_host();

        tensor_D_host.at(i).sync_host();

        cudaDeviceSynchronize();

        cutlass::TensorView<ElementC, LayoutC> view_D(tensor_D_host.at(i).host_ref(), problem.filter_extent());
        cutlass::TensorView<ElementC, LayoutC> view_Ref(tensor_d.host_ref(), problem.filter_extent());


        std::cout << type_name<decltype(view_D)>() << std::endl;
        std::cout << "Output (Size=" << view_D.size()  << ")" << std::endl;
        std::cout << std::endl;
        
        int start = view_D.size() - 2000;

        for (int idx = 0; idx < 10; ++idx) {
            std::cout << "Output \t\t[ " << start + idx*100 << " - " << start + idx*100 + 99 << " ] ";
            for (int offset = 0; offset < 100; ++offset) {
                std::cout << view_D.ref().data()[start + idx*100 + offset] << "\t";
            }
            std::cout << std::endl;
            std::cout << "Reference \t[ " << start + idx*100 << " - " << start + idx*100 + 99 << " ] ";
            for (int offset = 0; offset < 100; ++offset) {
                std::cout << view_Ref.ref().data()[start + idx*100 + offset] << "\t";
            }
            std::cout << std::endl;
        }
        

        // Reference check
        passed = cutlass::reference::host::TensorEquals(view_D, view_Ref);

        if (!passed) {
            std::cerr << "\n***\\\\\\nError - problem " << i << " failed the QA check\n***\n" << std::endl;
            // return passed;
        }
        if (passed) {
            std::cout << "Passed!!" << std::endl;
        }
    }

    return passed;
}

public:

/// Returns the number of threadblocks to launch if the kernel can run on the target
/// device. Otherwise, returns zero.
int sufficient() const {
    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    int occupancy = ImplicitGemmConvolution::maximum_active_blocks();

    return properties.multiProcessorCount * occupancy;

}


/// Executes a Grouped GEMM kernel and measures runtime.
Result profile_grouped() {

    Result result;

    int threadblock_count = sufficient();

    // Early exit
    if (!threadblock_count) {
    std::cout << "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel." << std::endl;
    return result;
    }

    if (options.verbose) {
    print_problem_sizes_();
    }

    result.passed = false;

    // Initialize the problem
    initialize_();

    // Configure the GEMM arguments
    typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

    // Configure GEMM arguments
    typename ImplicitGemmConvolution::Arguments args(
    problem_sizes_device.get(),
    problem_count(),
    threadblock_count,
    epilogue_op,
    ref_A.get(),
    ref_B.get(),
    ref_C.get(),
    ref_D.get(),
    &options.problem_sizes[0]
    );

    // Initialize the GEMM object
    ImplicitGemmConvolution implicit_gemm_convolution;

    size_t workspace_size = implicit_gemm_convolution.get_workspace_size(args);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    result.status = implicit_gemm_convolution.initialize(args, workspace.get());

    if (result.status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize CUTLASS Grouped ImplicitGemmConvolution kernel." << std::endl;
    std::cerr << cutlassGetStatusString(result.status) << std::endl;
    return result;
    }

    // Run the grouped GEMM object
    result.status = implicit_gemm_convolution.run();

    if (result.status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to run CUTLASS Grouped ImplicitGemmConvolution kernel." << std::endl;
    return result;
    }

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
    std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error) << std::endl;
    return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (options.reference_check) {
    result.passed = verify_();
    }

    //
    // Warm-up run of the grouped GEMM object
    //
    result.status = implicit_gemm_convolution.run();

    if (result.status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
    return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
    }
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {
    implicit_gemm_convolution();
    }

    //
    // Stop profiling loop
    //
    
    // Record an event when the GEMM operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
    (void)cudaEventDestroy(event);
    }
    


        
    return result;
}
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

//
// This example uses mma.sync to directly access Tensor Cores to achieve peak performance.
//

cudaDeviceProp props;

cudaError_t error = cudaGetDeviceProperties(&props, 0);
if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
}

// if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
// 
//     //
//     // This example requires an NVIDIA Ampere-architecture GPU.
//     //
// 
//     std::cout 
//     << "CUTLASS's Grouped GEMM example requires a GPU of NVIDIA's Ampere Architecture or "
//     << "later (compute capability 80 or greater).\n";
// 
//     return 0;
// }

//
// Parse options
//

Options options;

options.parse(argc, args);

if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
}

if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
}

//
// Define the Grouped GEMM type
//

using ElementInput = float;
using ElementOutput = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::TensorNHWC;
using LayoutB = cutlass::layout::TensorNHWC;
using LayoutC = cutlass::layout::TensorNHWC;

static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

using ConvKernel = typename cutlass::conv::kernel::DefaultConv2dWgradGrouped<
    ElementInput, 
    LayoutA,
    ElementInput,
    LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator, 
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 2>,
    cutlass::gemm::GemmShape<32, 32, 2>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    3,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm
    >::Conv2dWgradKernel;

using ImplicitGemmConvolutionGrouped = cutlass::conv::device::ImplicitGemmConvolutionGrouped<ConvKernel>;


//
// Profile it
//

TestbedGrouped<ImplicitGemmConvolutionGrouped> testbed1(options);
// TestbedGrouped<GemmGrouped8> testbed8(options);

if (!testbed1.sufficient()) {
    std::cout << "The active CUDA device lacks sufficient hardware resources to execute this kernel.\n";
    return 0;
}

Result result1 = testbed1.profile_grouped();
if (!result1.passed) {
    std::cout << "Profiling CUTLASS grouped GEMM has failed.\n";
    std::cout << "\nFailed\n";
    return -1;
}
// Result result8 = testbed8.profile_grouped();
// if (!result8.passed) {
//   std::cout << "Profiling CUTLASS grouped GEMM has failed.\n";
//   std::cout << "\nFailed\n";
//   return -1;
// }

std::vector<Result> result_list = {result1}; //, result2, result3, result4, result5, result6, result7, result8};  
double min_runtime = 1000;
int min_algo = 0;
for (int i=0; i<result_list.size(); i++){
    if (min_runtime > result_list[i].runtime_ms) {
    min_algo = i;
    }
    min_runtime = std::min(min_runtime, result_list[i].runtime_ms);
}

std::cout << "min_runtime " << min_runtime << " , min_algo " << min_algo << std::endl;

    std::ofstream writeFile(std::string(std::getenv("PWD")) + "/" + "result.txt");
std::cout << "write on " << std::string(std::getenv("PWD")) + "/" + "result.txt" << std::endl;
    if( 1){ // writeFile.is_open() ){
        writeFile << min_runtime;
        writeFile.close();
    }

return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

