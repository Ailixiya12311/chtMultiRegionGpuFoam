#!/bin/bash
# M8: OpenFOAM functionObjects run inside chtMultiRegionGpuFoam. The M2 case (V5 conjugate cavity, hot face step,
# deltaT 2 s, 20 s, 15 outer correctors) with probes (air and solid), fieldMinMax, volFieldValue and solverInfo in
# controlDict; OpenFOAM chtMultiRegionFoam and chtMultiRegionGpuFoam (BACKEND build) on copies of the same case,
# their postProcessing/ outputs compared column by column. The probes see every time step, so they show whether
# the fields OpenFOAM holds are the current ones (they used to be refreshed at write times only).
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-omp}
END=${ENDT:-20}
W=/w/cases/m8_fo
REF=/py/cases/cht_frame
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
rm -rf $W; mkdir -p $W
mk() {
  C=$W/$1; mkdir -p $C; cp -r $REF/0 $REF/constant $REF/system $C/
  python3 /py/verify/make_transient_v5.py $C $END 10 15 > /dev/null
  rm -rf $C/0; cp -r $REF/4000 $C/; python3 /py/verify/make_step_v5.py $C > /dev/null
  cat >> $C/system/controlDict <<'EOF'

functions
{
    probesAir
    {
        type probes; libs (sampling); region air;
        fields (T U p_rgh);
        probeLocations ((0.05 0.05 0.005) (0.07 0.10 0.005) (0.115 0.07 0.005));
    }
    probesSolid
    {
        type probes; libs (sampling); region solidL;
        fields (T);
        probeLocations ((0.01 0.07 0.005));
    }
    minMaxAir
    {
        type fieldMinMax; libs (fieldFunctionObjects); region air;
        fields (T U); mode magnitude; location false;
    }
    avgSolidT
    {
        type volFieldValue; libs (fieldFunctionObjects); region solidT;
        fields (T); operation volAverage; writeFields false;
    }
    residualsAir
    {
        type solverInfo; libs (utilityFunctionObjects); region air;
        fields (U p_rgh h);
    }
}
EOF
}
mk of; mk gpu
(cd $W/of && chtMultiRegionFoam > log.run 2>&1; echo "OpenFOAM rc=$? $(grep -c '^Time =' log.run) steps") &
(cd $W/gpu && /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; echo "gpu rc=$? $(grep -c '^Time =' log.run) steps") &
wait
grep -E "FOAM FATAL|FOAM Warning|gpucht FATAL" -A8 $W/of/log.run $W/gpu/log.run | head -40
python3 - "$W/of/postProcessing" "$W/gpu/postProcessing" <<'PY'
import os, re, sys
of, gp = sys.argv[1], sys.argv[2]
num = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
def rows(p):
    out = []
    for line in open(p):
        if line.lstrip().startswith("#") or not line.strip():
            continue
        out.append([float(x) for x in num.findall(line.replace("(", " ").replace(")", " "))])
    return out
worst = {}
for root, _, files in os.walk(of):
    for f in sorted(files):
        a = os.path.join(root, f)
        rel = os.path.relpath(a, of)
        b = os.path.join(gp, rel)
        if not os.path.exists(b):
            print(f"{rel}: MISSING in gpu"); continue
        ra, rb = rows(a), rows(b)
        if len(ra) != len(rb):
            print(f"{rel}: {len(ra)} rows (OpenFOAM) vs {len(rb)} rows (gpu)")
        n = min(len(ra), len(rb))
        d = 0.0
        for x, y in zip(ra[:n], rb[:n]):
            if len(x) != len(y):
                d = float("nan"); break
            d = max([d] + [abs(p - q) for p, q in zip(x[1:], y[1:])])   # column 0: time
        print(f"{rel}: {n} rows, max |gpu - OpenFOAM| over all columns = {d:.3e}")
        if rb:
            print(f"    last OpenFOAM: {ra[n-1]}\n    last gpu:      {rb[n-1]}")
PY
echo M8_DONE
