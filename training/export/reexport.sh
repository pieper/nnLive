#!/usr/bin/env bash
# Re-export the nnInteractive teacher at a browser-affordable patch size and re-split into WebGPU chunks.
# Runs entirely on CPU (CUDA_VISIBLE_DEVICES="") so it never touches the training GPU. Fully convolutional
# network -> exporting at 128^3 is math-equivalent to 192^3, just a smaller (memory-safe) field of view.
#   NNLIVE_TAG=128 NNLIVE_SIZE=128 bash reexport.sh
set -ex
export NNLIVE_TAG=${NNLIVE_TAG:-128}
export NNLIVE_SIZE=${NNLIVE_SIZE:-128}
export NNLIVE_DEV=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
source ~/nnlive-venv/bin/activate

python ~/export/export_onnx.py
python ~/export/replace_avgpool_slices.py
python ~/export/replace_convtranspose.py
python ~/export/split_model.py \
    --src ~/nnlive_export/net_${NNLIVE_TAG}_webgpu_fp16.onnx \
    --out ~/nnlive_chunks_${NNLIVE_TAG} --chunks 16
echo "REEXPORT_DONE tag=${NNLIVE_TAG} size=${NNLIVE_SIZE}"
