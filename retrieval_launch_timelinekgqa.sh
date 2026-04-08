export CUDA_VISIBLE_DEVICES=0
file_path=corpus
#index_file=$file_path/timequestions_kg_index/e5_Flat.index
#corpus_file=$file_path/timequestions_kg.jsonl
index_file=$file_path/icews_actor_kg_index/e5_Flat.index
corpus_file=$file_path/icews_actor_kg.jsonl

retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 30 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            --port 8080
  
