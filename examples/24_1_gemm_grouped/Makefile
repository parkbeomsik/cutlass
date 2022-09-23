CC := nvcc

INCLUDE_DIRS := -I/home/beomsik/dp/cutlass/include -I/home/beomsik/dp/cutlass/tools/util/include

TARGET := 24_1_gemm_grouped

all:
	$(CC) $(INCLUDE_DIRS) -O3 gemm_grouped.cu -o $(TARGET)