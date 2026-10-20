#!/bin/bash
# inside the OpenFOAM container: build the library (BACKEND omp|cuda) and the application
set -e
B=${BACKEND:-omp}
export GCOF=${GCOF:-/w} GCOF_BIN=${GCOF:-/w}/build/$B
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true
cd $GCOF/lib && make BACKEND=$B ARCH=${ARCH:-native} -j${JOBS:-6} all test 2>&1 | grep -E "error|Error" -A3 | head -40
rm -f $GCOF_BIN/chtMultiRegionGpuFoam
cd $GCOF/app/chtMultiRegionGpuFoam && wmake 2>&1 | grep -E "error|Error|warning: unused" -A3 | head -60
ls -la $GCOF_BIN/libgpucht.so $GCOF_BIN/test_lib $GCOF_BIN/chtMultiRegionGpuFoam
