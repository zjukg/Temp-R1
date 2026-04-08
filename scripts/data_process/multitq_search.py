# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MultiTQ dataset to parquet format
"""

import json
import os
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
import argparse



def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
#         prefix = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
#     else:
#         raise NotImplementedError
#     return prefix
    
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
  
- Example: "Which country did Obama visit right after his first visit to France?"
  → Q1: When was Obama's first visit to France?
  → Q2: What did Obama visit after that date?
  → Q3: Which one is chronologically next?

### Answer Format Specification
Match question wording exactly:
- "which month" / "what month" → **yyyy-mm** (e.g., 2009-06)
- "when" / "what date" → **yyyy-mm-dd** (e.g., 2009-06-15)  
- "which year" / "what year" → **yyyy** (e.g., 2009)
- "who" / "where" / "what" → **Entity name or concise phrase**

## Complete Examples

### Example 1: Simple Factual Question

**Question**: When did Obama visit Germany in 2009?

<plan>
- Question type: Simple factual
- Time constraints: Year 2009 (explicit scope)
- Sub-questions: None needed
- Answer format: yyyy-mm-dd
</plan>

<search>Obama visit Germany 2009</search>

<information>
Doc 1: Obama Make_a_visit Germany on 2009-06-05
</information>

<answer>2009-06-05</answer>

---

### Example 2: Multi-hop with Temporal Constraints

**Question**: Which country did Obama visit right after his first visit to France?

<plan>
- Question type: Multi-hop, Sequential temporal
- Time constraints: "first visit" (comparative - need earliest), "right after" (sequential - need immediate next)
- Sub-questions:
  1. When was Obama's first visit to France?
  2. What visits occurred after that date?
  3. Which one is chronologically next?
- Answer format: Country name
</plan>

<think>Need to find Obama's first France visit date first.</think>

<search>Obama visit France</search>

<information>
Doc 1: Obama Make_a_visit France on 2009-06-06
Doc 2: Obama Make_a_visit France on 2014-06-05
</information>

<think>First visit was 2009-06-06. Now need to find visits after this date.</think>

<search>Obama visit 2009</search>

<information>
Doc 1: Obama Make_a_visit Germany on 2009-06-05
Doc 2: Obama Make_a_visit France on 2009-06-06
Doc 3: Obama Make_a_visit Italy on 2009-06-10
Doc 4: Obama Make_a_visit Russia on 2009-07-06
</information>

<filter>
Doc 3: Obama Make_a_visit Italy on 2009-06-10
Doc 4: Obama Make_a_visit Russia on 2009-07-06
</filter>

<rank>
Doc 3: Obama Make_a_visit Italy on 2009-06-10
Doc 4: Obama Make_a_visit Russia on 2009-07-06
</rank>

<think>The chronologically next visit after France (2009-06-06) is Italy (2009-06-10).</think>

<answer>Italy</answer>

---

### Example 3: Comparative Temporal Question

**Question**: What is the earliest visit Obama made to Asia in 2009?

<plan>
- Question type: Comparative temporal
- Time constraints: "earliest" (comparative - need minimum date), "in 2009" (temporal scope)
- Sub-questions:
  1. What Asian countries did Obama visit in 2009?
  2. Which visit has the earliest date?
- Answer format: Country name or visit description
</plan>

<search>Obama visit Asia 2009</search>

<information>
Doc 1: Obama Make_a_visit Japan on 2009-11-13
Doc 2: Obama Make_a_visit China on 2009-11-15
Doc 3: Obama Make_a_visit South_Korea on 2009-11-19
</information>

<rank>
Doc 1: Obama Make_a_visit Japan on 2009-11-13
Doc 2: Obama Make_a_visit China on 2009-11-15
Doc 3: Obama Make_a_visit South_Korea on 2009-11-19
</rank>

<think>After sorting by date, Japan (2009-11-13) is the earliest.</think>

<answer>Japan</answer>

---

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

def load_multitq_data(json_path):
    """Load MultiTQ JSON data"""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


def process_multitq_example(example, idx, split, template_type, data_source):
    """Process a single MultiTQ example"""
    # Clean question
    question = example['question'].strip()
    if question and question[-1] != '?':
        question += '?'
    
    # Update the example with cleaned question
    example['question'] = question
    
    # Create prompt with template
    prompt_text = make_prefix(example, template_type=template_type)
    
    # Prepare solution with answers
    solution = {
        "target": example['answers'],  # MultiTQ uses 'answers' field (list)
    }
    
    # Create the data structure
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": prompt_text,
        }],
        "ability": "temporal-reasoning",  # MultiTQ is temporal reasoning
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'quid': example.get('quid'),
            'answer_type': example.get('answer_type'),
            'time_level': example.get('time_level'),
            'qtype': example.get('qtype'),
            'qlabel': example.get('qlabel'),
        }
    }
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/data/MultiTQ', 
                        help='Directory containing train_MULTITQ.json and dev.json')
    parser.add_argument('--local_dir', default='./data/multitq_search2',
                        help='Output directory for parquet files')
    parser.add_argument('--hdfs_dir', default=None,
                        help='Optional HDFS directory to copy files to')
    parser.add_argument('--template_type', type=str, default='base',
                        help='Template type for prompt formatting')
    
    args = parser.parse_args()
    
    data_source = 'multitq'
    
    # Create output directory if it doesn't exist
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Load train and dev data
    train_path = os.path.join(args.input_dir, 'train_MULTITQ.json')
    test_path = os.path.join(args.input_dir, 'test.json')
    
    print("="*70)
    print("Loading MultiTQ dataset...")
    print("="*70)
    
    train_data = load_multitq_data(train_path)
    test_data = load_multitq_data(test_path)
    
    # Process train data
    print("\nProcessing train data...")
    train_processed = []
    train_single = []  # 简单问题
    train_multiple = []  # 困难问题
    for idx, example in enumerate(train_data):
        processed = process_multitq_example(
            example, idx, 'train', args.template_type, data_source
        )
        train_processed.append(processed)
        
        # 根据qlabel分类
        if example.get('qlabel') == 'Single':
            train_single.append(processed)
        elif example.get('qlabel') == 'Multiple':
            train_multiple.append(processed)
        
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(train_data)} train examples")

    # Process dev data
    print("\nProcessing test data...")
    test_processed = []
    test_single = []
    test_multiple = []
    for idx, example in enumerate(test_data):
        processed = process_multitq_example(
            example, idx, 'test', args.template_type, data_source
        )
        test_processed.append(processed)
        
        if example.get('qlabel') == 'Single':
            test_single.append(processed)
        elif example.get('qlabel') == 'Multiple':
            test_multiple.append(processed)
        
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(test_data)} test examples")
    # Convert to HuggingFace Dataset
    print("\nConverting to HuggingFace Dataset format...")
    # 完整数据集
    train_dataset = Dataset.from_list(train_processed)
    test_dataset = Dataset.from_list(test_processed)

    # 简单问题数据集
    train_single_dataset = Dataset.from_list(train_single)
    test_single_dataset = Dataset.from_list(test_single)

    # 困难问题数据集
    train_multiple_dataset = Dataset.from_list(train_multiple)
    test_multiple_dataset = Dataset.from_list(test_multiple)

    print("\nSaving parquet files...")
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train_all.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test_all.parquet'))

    train_single_dataset.to_parquet(os.path.join(args.local_dir, 'train_single.parquet'))
    test_single_dataset.to_parquet(os.path.join(args.local_dir, 'test_single.parquet'))

    train_multiple_dataset.to_parquet(os.path.join(args.local_dir, 'train_multiple.parquet'))
    test_multiple_dataset.to_parquet(os.path.join(args.local_dir, 'test_multiple.parquet'))

    # 输出统计信息
    print(f"\n{'='*70}")
    print("Dataset Statistics:")
    print(f"{'='*70}")
    print(f"Train - All: {len(train_dataset)}")
    print(f"Train - Single: {len(train_single_dataset)} ({len(train_single_dataset)/len(train_dataset)*100:.1f}%)")
    print(f"Train - Multiple: {len(train_multiple_dataset)} ({len(train_multiple_dataset)/len(train_dataset)*100:.1f}%)")
    print(f"\nDev - All: {len(test_dataset)}")
    print(f"Dev - Single: {len(test_single_dataset)} ({len(test_single_dataset)/len(test_dataset)*100:.1f}%)")
    print(f"Dev - Multiple: {len(test_multiple_dataset)} ({len(test_multiple_dataset)/len(test_dataset)*100:.1f}%)")
    print(f"{'='*70}")
    # Optional: copy to HDFS
    if args.hdfs_dir is not None:
        print(f"\nCopying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print("✓ Copied to HDFS")
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)