import json
import os
from tqdm import tqdm

# 配置输入和输出路径
input_file = "/corpus/full.txt"
output_dir = "/corpus"
output_file = os.path.join(output_dir, "full_kg.jsonl")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print(f"Converting {input_file} to {output_file}...")

with open(input_file, 'r', encoding="utf-8") as fin, open(output_file, 'w', encoding="utf-8") as fout:
    count = 0
    for idx, line in enumerate(tqdm(fin, desc="Converting triples")):
        line = line.strip()

        # 跳过空行或注释
        if not line or line.startswith('#'):
            continue

        parts = [part.strip() for part in line.split('|') if part.strip()]

        # kb.txt 中每一行都应该是 subject|predicate|object，如果不满足则跳过
        if len(parts) < 3:
            continue

        subject, predicate, obj = parts[:3]

        # 将谓词中的下划线替换成空格，使描述更加自然
        predicate_text = predicate.replace('_', ' ')
        content = f"{subject} {predicate_text} {obj}"

        json_obj = {
            "id": str(idx),
            "contents": content
        }

        fout.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        count += 1

print(f"Conversion complete! Processed {count} lines.")
