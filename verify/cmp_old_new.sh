source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true
cd /w
for p in m1_n30/gpu m3_n15_t0/pas_gpu m3_n15_t0/cp_gpu m3_n15_t0/full_gpu m4_n20_t0/gpu m5_ftc60/gpu m5_v7/gpu m6_adj/gpu m6_v7full/gpu m7_v7full/cuda m7_v7full/omp; do
  N=cases/$p; O=cases_before_20261020/$p
  [ -d $N ] && [ -d $O ] || { echo "$p: missing"; continue; }
  if [ $(find $N -maxdepth 1 -name log.run -newer cases_before_20261020/m1_n30 | wc -l) = 0 ]; then echo "$p: not re-run"; continue; fi
  t=$(ls -d $N/[0-9]* | xargs -n1 basename | sort -g | tail -1)
  to=$(ls -d $O/[0-9]* | xargs -n1 basename | sort -g | tail -1)
  echo "== $p  new t=$t  old t=$to  ($(grep -m1 -o 'build/[a-z]*' $O/log.run 2>/dev/null) -> $(grep -m1 -o 'build/[a-z]*' $N/log.run 2>/dev/null))"
  python3 /w/verify/compare_of.py $N $t $O $to /tmp/cmp.json 2>&1 | awk '{print "   " $0}' | sort -t' ' -k3 | head -14
done
