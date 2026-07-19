#!/usr/bin/env bash
# Real-data distillation (up to 8h, plateau early-stop) -> export browser-native ONNX ->
# publish weights to the CORS-enabled Ceph bucket -> stop the filler -> shelve the instance.
# The cache filler is launched separately (runs concurrently).
#
# SAFETY: the instance is shelved ONLY on a genuine completion — distill_real must exit 0 AND leave
# its DONE sentinel, AND the weights must have published to the bucket. If distill is killed/crashes,
# or the upload fails, we leave the box UP for inspection / retry. (An earlier version shelved
# unconditionally, so killing distill_real mid-run raced the script straight into the shelve step.)
set -x
OUT=~/nnlive_student_real; mkdir -p "$OUT"
CLOUD=MED250016_IU
CONTAINER=nnlive-models
SWIFT=https://js2.jetstream-cloud.org:8001/swift/v1
rm -f "$OUT/DONE"                                # distill_real writes this only on real completion
source ~/nnlive-venv/bin/activate

python ~/distill_real.py --max_hours "${MAX_HOURS:-8}" --out "$OUT"
status=$?
echo "distill exit=$status $(date)"

if [ "$status" -ne 0 ] || [ ! -f "$OUT/DONE" ]; then
    echo "distill did NOT complete normally (status=$status, DONE=$([ -f "$OUT/DONE" ] && echo yes || echo no)) -- leaving instance UP, NOT shelving"
    exit "$status"
fi

python ~/export_student.py --ckpt "$OUT/student.pt" --out "$OUT/student_real_128" || echo "EXPORT FAILED"

# ---- publish weights to the CORS-enabled Ceph bucket (retry; must succeed before we shelve) ----
publish() {
    local f="$1" name="$2" ctype="$3"
    [ -f "$f" ] || { echo "publish: missing $f"; return 1; }
    for try in 1 2 3; do
        local tok; tok=$(openstack --os-cloud="$CLOUD" token issue -c id -f value 2>/dev/null)
        local code; code=$(curl -s -o /dev/null -w "%{http_code}" -X PUT -H "X-Auth-Token: $tok" \
            -H "Content-Type: $ctype" --data-binary @"$f" "$SWIFT/$CONTAINER/$name")
        echo "publish $name try$try -> $code"
        case "$code" in 201|202) return 0;; esac
        sleep 5
    done
    return 1
}
pub_ok=1
publish "$OUT/student_real_128_fp16.onnx" "student_real_128_fp16.onnx" "application/octet-stream" || pub_ok=0
publish "$OUT/DONE" "student_real_128.meta.txt" "text/plain" || echo "meta upload failed (non-fatal)"

if [ "$pub_ok" -ne 1 ]; then
    echo "WEIGHTS DID NOT PUBLISH to $SWIFT/$CONTAINER -- leaving instance UP so we can retry (NOT shelving)"
    exit 1
fi
echo "PUBLISHED: $SWIFT/$CONTAINER/student_real_128_fp16.onnx"

pkill -f cache_filler.py 2>/dev/null            # stop the producer (if still running)
UUID=$(curl -s http://169.254.169.254/openstack/latest/meta_data.json \
       | python3 -c "import sys,json;print(json.load(sys.stdin)['uuid'])")
echo "SHELVING $UUID $(date)"
openstack --os-cloud="$CLOUD" server shelve "$UUID"
