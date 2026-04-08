import json
import os
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
import argparse

def make_prefix(dp, template_type):
    question = dp['question']
    if template_type == 'base':
        prefix = f"""You are a question-answering assistant with access to a knowledge base.

## Response Structure

**REQUIRED FIRST STEP**: Start with planning
<plan>
- Question type: [问题类型]
- Time constraints: [时间约束关键词]
- Sub-questions: [子问题列表]
- Answer format: [答案格式]
</plan>

Then use tools in any order that makes sense:

### 🧠 Reasoning Tool
**`<think>your reasoning</think>`**
- Analyze information, explain logic, evaluate results
- Can use multiple times throughout

### 🔍 Search Tool  
**`<search>query</search>`**
- Search the knowledge base
- Returns up to 20 facts in format: "Doc X: Entity Relation Entity on Date"
- **`<information>facts</information>`** - Auto-filled by system

### 📋 Processing Tools
**`<filter>facts</filter>`**
- Filter by semantic relevance or temporal constraints
- Preserve format: "Doc X: Entity Relation Entity on Date"

**`<rank>facts</rank>`**
- Sort facts by date (ascending/descending)
- Preserve format: "Doc X: Entity Relation Entity on Date"

### ✅ Answer Tool
**`<answer>final answer</answer>`**
- Provide final answer when confident
- Must match the format specified in your plan

## Planning Guidelines

### Question Type Categories
- **Simple factual**: Direct query answerable with one search
- **Multi-hop**: Requires multiple steps or information pieces
- **Comparative temporal**: Involves "first", "last", "earliest", "latest"
- **Sequential temporal**: Involves "after", "before", "right after", "next"
- **Temporal range**: Involves "between", "during", "in [year]"
- **Duration/Interval**: Calculating length of time or finding overlaps

### Time Constraints to Identify
- **Sequential words**: "after", "before", "right after", "immediately before", "next", "previous"
- **Comparative words**: "first", "last", "earliest", "latest", "most recent", "oldest"
- **Range indicators**: "between X and Y", "during", "in [year]", "from X to Y"
- **Implicit references**: "since then", "afterwards", "previously"

### Sub-question Decomposition
For multi-hop questions, break into ordered steps:
- Example: "Where did X's spouse work?"
  → Q1: Who is X's spouse?
  → Q2: Where did that person work?

### Answer Format Specification
Match question wording exactly:
- "which month" / "what month" → **yyyy-mm** (e.g., 2009-06)
- "when" / "what date" → **yyyy-mm-dd** (e.g., 2009-06-15)  
- "which year" / "what year" → **yyyy** (e.g., 2009)
- "how long" / "duration" → **X years, Y months...** (Match the target format exactly)
- "who" / "where" / "what" → **Entity name or concise phrase**

## Important Constraints

1. **Every response MUST start with `<plan>`**
2. **Answer format MUST match the format specified in your plan**
3. **Only use facts from retrieved information** - no fabrication
4. **Preserve exact fact format** when using filter/rank
5. **End with `<answer>` tag** containing only the final answer

Now, please answer the following question: {question}
"""
    else:
        raise NotImplementedError
    return prefix

def load_timelinekgqa_data(json_path):
    print(f"Loading TimelineKGQA from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data

def is_valid_answer(ans):
    if ans is None:
        return False
    if isinstance(ans, str):
        if ans.strip() == "":
            return False
    if isinstance(ans, list):
        if len(ans) == 0:
            return False
    return True

def process_timeline_example(example, idx, split, template_type, data_source):
    if not is_valid_answer(example.get("answer")):
        return None
    q = example["question"].strip()
    if q and q[-1] != "?":
        q += "?"
    example["question"] = q

    prompt_text = make_prefix(example, template_type=template_type)
    solution = {"target": example["answer"]}

    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "temporal-reasoning",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": split,
            "index": idx,
            "qtype": example.get("question_type"),
            "qlabel": example.get("question_level"),
            "answer_type": example.get("answer_type"),
        }
    }
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default="/data/TimelineKGQA/unified_kg_icews_actor_questions_all.json", help="TimelineKGQA JSON file")
    parser.add_argument("--local_dir", default="/data/timelinekgqa_icews_actor_search", help="Output directory")
    parser.add_argument("--template_type", type=str, default="base")
    args = parser.parse_args()

    data_source = "timelinekgqa"
    os.makedirs(args.local_dir, exist_ok=True)

    print("=" * 70)
    data = load_timelinekgqa_data(args.input_json)
    
    # 1. 基础过滤
    data = [x for x in data if is_valid_answer(x.get("answer"))]
    
    # 2. 初始化分类字典
    # 使用 'complex' 替换之前的 'hard'
    levels = ['simple', 'medium', 'complex', 'all']
    splits = ['train', 'test']
    dataset_bins = {s: {l: [] for l in levels} for s in splits}

    print("\nProcessing and Categorizing data...")
    for idx, example in enumerate(data):
        split = example.get("split", "train")
        if split not in splits:
            continue
            
        # 统一转小写进行匹配
        raw_level = str(example.get("question_level", "simple")).lower()
        
        # 核心修改：匹配逻辑
        if 'simple' in raw_level:
            target_level = 'simple'
        elif 'medium' in raw_level:
            target_level = 'medium'
        elif 'complex' in raw_level: # 识别 complex
            target_level = 'complex'
        else:
            target_level = 'simple' 

        processed = process_timeline_example(example, idx, split, args.template_type, data_source)
        
        if processed:
            dataset_bins[split][target_level].append(processed)
            dataset_bins[split]['all'].append(processed)

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} examples...")

    # 3. 保存文件
    print("\nSaving Parquet files...")
    stats = []
    for s in splits:
        for l in levels:
            subset = dataset_bins[s][l]
            if len(subset) > 0:
                hf_dataset = Dataset.from_list(subset)
                # 文件名例如: train_complex.parquet
                filename = f"{s}_{l}.parquet"
                save_path = os.path.join(args.local_dir, filename)
                hf_dataset.to_parquet(save_path)
                stats.append(f"{s.capitalize()} - {l.capitalize()}: {len(subset)}")
            else:
                stats.append(f"{s.capitalize()} - {l.capitalize()}: 0 (Empty)")

    print("\n" + "=" * 70)
    print("Dataset Statistics:")
    print("=" * 70)
    for line in stats:
        print(line)
    print("=" * 70)
    print(f"All files saved to: {args.local_dir}")