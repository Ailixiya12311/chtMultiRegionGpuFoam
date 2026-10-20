# Compute library libgpucht.so
#   make BACKEND=omp            g++ -fopenmp, Thrust OMP backend (NAS, development and CPU runs)
#   make BACKEND=cuda ARCH=sm_80  nvcc, CUDA backend (Thrust/CUB of the CUDA toolkit)
BACKEND ?= omp
ARCH ?= native
CCCL ?= $(abspath ../third_party/cccl-3.0.3)
OUT ?= $(abspath ../build/$(BACKEND))
SRC = core.cpp mesh.cpp ldu.cpp linsolve.cpp energy.cpp flow.cpp building.cpp engine.cpp
OBJ = $(addprefix $(OUT)/,$(SRC:.cpp=.o))

ifeq ($(BACKEND),omp)
CXX = g++
CXXFLAGS = -O3 -std=c++17 -fPIC -fopenmp -Wall -Wno-subobject-linkage \
           -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP \
           -I$(CCCL)/thrust -I$(CCCL)/libcudacxx/include -I$(CCCL)/cub
LDFLAGS = -shared -fopenmp
else
CXX = nvcc
ifeq ($(ARCH),all)
NVARCH = -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_89,code=sm_89 \
         -gencode arch=compute_90,code=sm_90 \
         -gencode arch=compute_120,code=sm_120 \
         -gencode arch=compute_120,code=compute_120
else
NVARCH = -arch=$(ARCH)
endif
CXXFLAGS = -O3 -std=c++17 -x cu --extended-lambda $(NVARCH) -Xcompiler -fPIC
LDFLAGS = -shared
endif

all: $(OUT)/libgpucht.so

$(OUT)/%.o: %.cpp *.h
	@mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/libgpucht.so: $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ)

test: $(OUT)/test_lib

$(OUT)/test_lib: ../test/test_lib.cpp $(OUT)/libgpucht.so
	$(CXX) $(CXXFLAGS) -I. $< -o $@ -L$(OUT) -lgpucht -Xlinker -rpath='$$ORIGIN' -Xlinker -rpath=$(OUT) $(if $(filter omp,$(BACKEND)),-fopenmp,)

clean:
	rm -rf $(OUT)

.PHONY: all test clean
