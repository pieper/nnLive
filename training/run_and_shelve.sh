#!/usr/bin/env bash
# Train the student, export a browser-native ONNX, then SHELVE this instance to stop the
# allocation burn. Run detached:  STEPS=30000 nohup bash run_and_shelve.sh > ~/nnlive_student/run.log 2>&1 &
set -x
source ~/nnlive-venv/bin/activate
cd ~
OUT=~/nnlive_student
mkdir -p "$OUT"

echo "TRAIN START $(date)"
python distill.py --steps "${STEPS:-30000}" --patch 128 --bs 2 --base 32 --out "$OUT"
status=$?
echo "distill exit=$status $(date)"

if [ "$status" -eq 0 ]; then
    python export_student.py --ckpt "$OUT/student.pt" --out "$OUT/student_128" || echo "EXPORT FAILED"
fi

# Auto-shelve regardless of outcome (disk + logs are preserved through shelve; unshelve to resume/debug).
UUID=$(curl -s http://169.254.169.254/openstack/latest/meta_data.json \
       | python3 -c "import sys,json;print(json.load(sys.stdin)['uuid'])")
echo "SHELVING $UUID $(date)"
openstack --os-cloud=MED250016_IU server shelve "$UUID"
echo "SHELVE ISSUED $(date)"
