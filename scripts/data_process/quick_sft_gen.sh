
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tempr1

# ============ 1. 基础配置 ============
NUM_SAMPLES=1000
START_INDEX=0
USE_CORRECT_ONLY=true   # true=只保留正确的, false=保留所有

# ============ 2. API 配置 (OpenAI模式) ============
# 你的聚合API地址
BASE_URL="xxx"
# 你的 API Key
API_KEY="xxx"
MODEL_NAME="gpt-4o" 

# ============ 3. 检索器配置 ============
# 请确认你的检索器端口是 8000 还是 8080 (Python脚本默认是8000，这里写的是8080)
RETRIEVER_URL="http://127.0.0.1:8080/retrieve"


# 确保输出目录存在
mkdir -p outputs
mkdir -p logs

FILTER_FLAG=""
[ "$USE_CORRECT_ONLY" = true ] && FILTER_FLAG="--filter_correct_only"

MAX_WORKERS=20

python scripts/data_process/generate_sft_cron.py \
    --input /data/TimelineKGQA/unified_kg_cron_questions_train_balanced_1500.json \
    --output outputs/gpt4o-1000-cron-new.jsonl \
    --num_samples $NUM_SAMPLES \
    --start_index $START_INDEX \
    --max_workers $MAX_WORKERS \
    --model_api openai \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --retriever_url "$RETRIEVER_URL" \
    $FILTER_FLAG \
    2>&1 | tee logs/sft_gen_$(date +%Y%m%d_%H%M%S).log

