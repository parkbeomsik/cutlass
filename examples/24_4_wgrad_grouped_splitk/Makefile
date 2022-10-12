CC := nvcc

INCLUDE_DIRS := -I/home/beomsik/dp/cutlass/include -I/home/beomsik/dp/cutlass/tools/util/include

TARGET := wgrad_grouped_splitk

all:
	$(CC) $(INCLUDE_DIRS) -O3 wgrad_grouped_splitk.cu -o $(TARGET)