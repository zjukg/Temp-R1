cd /search_r1/search
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # 禁用点对点通信
export CUDA_LAUNCH_BLOCKING=1

# 设置参数
corpus_file=/corpus/full_kg.jsonl
save_dir=/corpus/kg_index
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
