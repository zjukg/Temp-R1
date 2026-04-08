#!/bin/bash
# Quick SFT data generation with minimal configuration

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tempr1

# ============ 快速配置（只需修改这里） ============
NUM_SAMPLES=1000
START_INDEX=0                                 # 生成数量
API_KEY="xxx"                         # 替换为你的 API Key
BASE_URL="xxx"              # OpenAI API Base URL
USE_CORRECT_ONLY=true                             # true=只保留正确的, false=保留所有


FILTER_FLAG=""
[ "$USE_CORRECT_ONLY" = true ] && FILTER_FLAG="--filter_correct_only"
# --model "$MODEL_URL" \
    # --api_key "$API_KEY" \
    # --base_url "$BASE_URL" \
python scripts/data_process/generate_sft_trajectories.py \
    --input /data/MultiTQ/oversampled_for_sft.json \
    --output /outputs/train_sft_oversampled_merged.jsonl \
    --num_samples $NUM_SAMPLES \
    --start_index $START_INDEX \
    --model_api openai \
    --retriever_url http://127.0.0.1:8000/retrieve \
    $FILTER_FLAG \
    2>&1 | tee logs/sft_gen_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Done!"