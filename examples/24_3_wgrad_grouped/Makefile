CC := nvcc

INCLUDE_DIRS := -I/home/beomsik/dp/cutlass/include -I/home/beomsik/dp/cutlass/tools/util/include

TARGET := wgrad_grouped

all:
	$(CC) $(INCLUDE_DIRS) -O3 wgrad_grouped.cu -o $(TARGET)