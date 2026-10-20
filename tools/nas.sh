#!/bin/bash
# Sync lib/ test/ app/ verify/ tools/ README.md Allwmake Allwclean to the NAS work dir and run a command in the isolated OpenFOAM container.
#   tools/nas.sh 'cd /w/lib && make BACKEND=omp test'        (from Git Bash on the Windows machine)
# CPUS / MEM env override the container limits (default 6 cores, 8 GB).
# Host, key and paths come from tools/nas.env (not committed; copy tools/nas.env.example).
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$P/tools/nas.env" ] || { echo "tools/nas.env missing (see tools/nas.env.example)" >&2; exit 1; }
. "$P/tools/nas.env"
W=$NAS_WORK/gpu_cht_of_r1
K="-i $NAS_KEY -o IdentitiesOnly=yes -o BatchMode=yes"
T=$(mktemp -u /tmp/gcof_XXXX).tar
(cd "$P" && tar cf "$T" --exclude=tools/nas.env $(ls -d lib test app verify tools README.md Allwmake Allwclean 2>/dev/null))
scp $K "$T" $NAS_HOST:/tmp/gcof_sync.tar >/dev/null
rm -f "$T"
ssh $K $NAS_HOST "cd $W && tar xf /tmp/gcof_sync.tar && chown -R 65532:65532 $W && \
  docker run --rm --gpus all --network none --read-only --user 65532:65532 --memory ${MEM:-8g} --cpus ${CPUS:-6} \
  --tmpfs /tmp:rw,size=2g --tmpfs /work/home:rw,size=256m -e HOME=/work/home -e TMPDIR=/tmp \
  -v $NAS_WORK/cuda-12.8:/usr/local/cuda:ro \
  -e PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  -v $W:/w -v $NAS_WORK/gpu_cht_solver_r1:/py:ro \
  --entrypoint bash $NAS_IMAGE -c '$1'"
