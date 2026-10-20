#!/bin/bash
# inside the OpenFOAM container: build the library (BACKEND omp|cuda) and the application into build/<backend>
# (the layout the verify/ scripts use; ./Allwmake is the normal install into the OpenFOAM user directories)
set -e
B=${BACKEND:-omp}
GCOF=${GCOF:-/w}
export GPUCHT_LIBBIN=$GCOF/build/$B GPUCHT_APPBIN=$GCOF/build/$B
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true
cd $GCOF/lib && make BACKEND=$B ARCH=${ARCH:-native} -j${JOBS:-6} all test 2>&1 | grep -E "error|Error" -A3 | head -40
rm -f $GPUCHT_APPBIN/chtMultiRegionGpuFoam
cd $GCOF/app/chtMultiRegionGpuFoam && wmake 2>&1 | grep -E "error|Error|warning: unused" -A3 | head -60
ls -la $GPUCHT_LIBBIN/libgpucht.so $GPUCHT_LIBBIN/test_lib $GPUCHT_APPBIN/chtMultiRegionGpuFoam
