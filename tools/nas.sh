#!/bin/bash
# Sync lib/ test/ app/ verify/ tools/ README.md to the NAS work dir and run a command in the isolated OpenFOAM container.
#   tools/nas.sh 'cd /w/lib && make BACKEND=omp test'        (from Git Bash on the Windows machine)
# CPUS / MEM env override the container limits (default 6 cores, 8 GB).
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
W=/path/to/work/gpu_cht_of_r1
K="-i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes"
T=$(mktemp -u /tmp/gcof_XXXX).tar
(cd "$P" && tar cf "$T" $(ls -d lib test app verify tools README.md 2>/dev/null))
scp $K "$T" user@gpu-host:/tmp/gcof_sync.tar >/dev/null
rm -f "$T"
ssh $K user@gpu-host "cd $W && tar xf /tmp/gcof_sync.tar && chown -R 65532:65532 $W && \
  docker run --rm --gpus all --network none --read-only --user 65532:65532 --memory ${MEM:-8g} --cpus ${CPUS:-6} \
  --tmpfs /tmp:rw,size=2g --tmpfs /work/home:rw,size=256m -e HOME=/work/home -e TMPDIR=/tmp \
  -v /path/to/work/cuda-12.8:/usr/local/cuda:ro \
  -e PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  -v $W:/w -v /path/to/work/gpu_cht_solver_r1:/py:ro \
  --entrypoint bash local/openfoam2606-cuda -c '$1'"
