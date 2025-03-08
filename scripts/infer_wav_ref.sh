cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES=0

python3 infer/infer.py \
    --lrc-path infer/example/eg_cn.lrc \
    --ref-audio-path infer/example/eg_cn.wav \
    --audio-length 95 \
    --repo_id ASLP-lab/DiffRhythm-base \
    --output-dir infer/example/output \
    --chunked