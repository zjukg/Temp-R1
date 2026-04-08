#!/usr/bin/env python3
"""
Generate high-quality trajectories for SFT training
使用模型API和检索器生成高质量的SFT训练轨迹
"""

import json
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import time
import re
from verl.utils.reward_score.qa_em_format import normalize_answer 

# ============ Instruction Template ============
INSTRUCTION = """You are a question-answering assistant with access to a knowledge base.

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

## Important Constraints

1. **Every response MUST start with `<plan>`**
2. **Answer format MUST match the format specified in your plan**
3. **Only use facts from retrieved information** - no fabrication
4. **Preserve exact fact format** when using filter/rank
5. **End with `<answer>` tag** containing only the final answer

Now, please answer the following question: {question}
"""

# ============ Retriever Functions ============
def query_retriever(question: str, retriever_url: str = "http://127.0.0.1:8000/retrieve", topk: int = 30) -> List[str]:
    """Query the retrieval service - matches generation.py format"""
    try:
        # ✅ 完全匹配 generation.py 的请求格式
        payload = {
            "queries": [question],      # 列表格式
            "topk": topk,               # 使用topk（不是top_k）
            "return_scores": True       # 返回scores（generation.py需要）
        }
        
        response = requests.post(
            retriever_url,
            json=payload,
            timeout=30  # 增加超时时间
        )
        response.raise_for_status()
        results = response.json()
        
        # ✅ 完全匹配 generation.py 的解析方式
        formatted = []
        
        if 'result' in results and len(results['result']) > 0:
            retrieval_result = results['result'][0]  # 取第一个query的结果
            
            for idx, doc_item in enumerate(retrieval_result, 1):
                try:
                    # ✅ 使用和 generation.py 相同的格式化逻辑
                    content = doc_item['document']['contents']
                    
                    # 分离标题和文本（如果有的话）
                    if "\n" in content:
                        title = content.split("\n")[0]
                        text = "\n".join(content.split("\n")[1:])
                        formatted.append(f"Doc {idx}(Title: {title}) {text}")
                    else:
                        # 如果没有换行符，直接使用content
                        formatted.append(f"Doc {idx}: {content}")
                        
                except (KeyError, TypeError) as e:
                    print(f"  ⚠️ Warning: Failed to parse doc {idx}: {e}")
                    # 尝试其他格式
                    if isinstance(doc_item, dict):
                        content = str(doc_item.get('document', doc_item))
                        formatted.append(f"Doc {idx}: {content}")
        
        if not formatted:
            print(f"  ⚠️ No results formatted from response")
            print(f"  Response keys: {results.keys()}")
            if 'result' in results:
                print(f"  Result length: {len(results['result'])}")
        
        return formatted
    
    except requests.exceptions.Timeout:
        print(f"⚠️  Retriever timeout after 30s")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"⚠️  Retriever HTTP error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Status code: {e.response.status_code}")
            print(f"  Response text: {e.response.text[:500]}")
        return []
    except Exception as e:
        print(f"⚠️  Retriever error: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_information(facts: List[str]) -> str:
    """Format facts as <information> block"""
    return "\n".join(facts)


# ============ Model API Functions ============
def call_openai_api(prompt: str, model: str = "gpt-4o", api_key: str = None, base_url: str = None, max_retries: int = 3) -> str:
    """Call OpenAI API or compatible API"""
    import openai
    import time
    
    if base_url:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = openai.OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            print(f"  🌐 API call attempt {attempt+1}/{max_retries}...", end='', flush=True)
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096,
                timeout=60  # ✅ 显式设置60秒超时
            )
            
            elapsed = time.time() - start_time
            print(f" Done in {elapsed:.1f}s")
            return response.choices[0].message.content
            
        except openai.APITimeoutError as e:
            print(f" ⏱️ Timeout after 60s")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"  ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ All retries failed")
                return None
                
        except openai.RateLimitError as e:
            print(f" 🚫 Rate limit hit")
            if attempt < max_retries - 1:
                wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                print(f"  ⏳ Waiting {wait_time}s for rate limit cooldown...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Rate limit persists")
                return None
                
        except Exception as e:
            print(f" ❌ Error: {e}")
            if attempt < max_retries - 1:
                print(f"  🔄 Retrying...")
                time.sleep(5)
            else:
                return None
    
    return None

def call_local_vllm(prompt: str, model_url: str = "http://localhost:8001/v1/chat/completions", model_name: str = "qwen") -> str:
    """Call local vLLM server"""
    try:
        response = requests.post(
            model_url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 4096
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"⚠️  Local model error: {e}")
        return None

def truncate_at_first_action(response: str) -> str:
    """Truncate response at first action end tag"""
    # 找到第一个结束标签
    search_pos = response.find('</search>')
    answer_pos = response.find('</answer>')
    
    # 选择最早出现的
    positions = [(search_pos, '</search>'), (answer_pos, '</answer>')]
    positions = [(pos, tag) for pos, tag in positions if pos != -1]
    
    if not positions:
        return response
    
    first_pos, first_tag = min(positions, key=lambda x: x[0])
    return response[:first_pos + len(first_tag)]

def extract_search_query(response: str) -> str:
    """Extract search query from <search> tags"""
    match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
    return match.group(1).strip() if match else ""

# ============ Interactive Trajectory Generation ============
def generate_trajectory_interactive(
    question: str,
    retriever_url: str = "http://127.0.0.1:8000/retrieve",
    topk: int = 30,
    model_api: str = "openai",
    **api_kwargs
) -> str:
    """
    Generate trajectory by interactively calling model and retriever
    模拟真实的检索增强生成过程
    """
    full_trajectory = ""
    conversation_history = INSTRUCTION.replace("{question}", question)
    max_turns = 3  # 增加轮次，允许多次搜索
    
    for turn in range(max_turns):
        print(f"  🔄 Turn {turn+1}/{max_turns}")
        
        # ✅ 调用模型API
        if model_api == "openai":
            response = call_openai_api(conversation_history, **api_kwargs)
        elif model_api == "local":
            response = call_local_vllm(conversation_history, **api_kwargs)
        else:
            raise ValueError(f"Unknown model_api: {model_api}")
        
        if not response:
            print(f"  ❌ Model API failed")
            break
        
        # ✅ 立即截断在第一个动作结束标签
        response = truncate_at_first_action(response)
        
        # 添加到轨迹
        full_trajectory += response
        
        # ✅ 检查动作类型并执行相应操作
        if '</search>' in response:
            # 提取search query
            search_query = extract_search_query(response)
            if not search_query:
                print(f"  ⚠️ Empty search query")
                break
            
            print(f"  🔍 Searching: {search_query[:60]}...")
            
            # 执行检索
            retrieved_facts = query_retriever(search_query, retriever_url, topk)
            
            if not retrieved_facts:
                print(f"  ⚠️ No retrieval results")
                information_block = "\n\n<information>\nNo relevant information found.\n</information>\n\n"
            else:
                print(f"  ✅ Retrieved {len(retrieved_facts)} facts")
                formatted_info = format_information(retrieved_facts)
                information_block = f"\n\n<information>\n{formatted_info}\n</information>\n\n"
            
            # 添加检索结果到轨迹
            full_trajectory += information_block
            
            # ✅ 更新conversation history（包含完整轨迹）
            conversation_history = INSTRUCTION.replace("{question}", question) + "\n\n" + full_trajectory
            
            # 继续下一轮，让模型基于检索结果继续
            
        elif '</answer>' in response:
            # 找到答案，完成生成
            print(f"  ✅ Answer generated")
            break
            
        elif '</think>' in response or '</plan>' in response:
            # 思考或规划步骤，继续下一轮
            conversation_history = INSTRUCTION.replace("{question}", question) + "\n\n" + full_trajectory
            
        elif '</filter>' in response or '</rank>' in response:
            # 过滤或排序步骤，继续下一轮
            conversation_history = INSTRUCTION.replace("{question}", question) + "\n\n" + full_trajectory
            
        else:
            # 无效响应或没有明确的结束标签
            print(f"  ⚠️ No valid action tag found in response")
            # 尝试继续，但如果连续失败就退出
            if turn > 0:  # 给第一轮一些容错
                break
            conversation_history = INSTRUCTION.replace("{question}", question) + "\n\n" + full_trajectory
    
    return full_trajectory

def enhanced_normalize(text: str) -> str:
    """
    Enhanced normalization for entity matching
    处理知识图谱格式 vs 人类可读格式的差异
    """
    # ✅ 先替换格式字符为空格（在normalize之前）
    format_chars = ['_', '(', ')', '/', '-', '.', ',', ';', ':', '|']
    for char in format_chars:
        text = text.replace(char, ' ')
    
    # 然后用标准normalize（处理大小写、冠词等）
    from verl.utils.reward_score.qa_em_format import normalize_answer
    text = normalize_answer(text)
    
    # 最后规范化空格（多个空格合并为一个）
    text = ' '.join(text.split())
    
    return text

# ============ Plan Validation Functions ============
def validate_plan(trajectory: str) -> Dict[str, Any]:
    """Validate plan quality"""
    results = {
        'has_plan': False,
        'plan_position_valid': False,
        'plan_complete': False,
        'missing_fields': [],
        'plan_content': None
    }
    
    # Check if has plan
    plan_match = re.search(r'<plan>(.*?)</plan>', trajectory, re.DOTALL)
    if not plan_match:
        return results
    
    results['has_plan'] = True
    plan_content = plan_match.group(1).strip()
    results['plan_content'] = plan_content
    
    # Check if plan is at the beginning (allow some whitespace)
    plan_start = trajectory.find('<plan>')
    if plan_start < 50:  # Allow some initial whitespace/newlines
        results['plan_position_valid'] = True
    
    # Check for required fields
    required_fields = {
        'Question type:': 'question_type',
        'Time constraints:': 'time_constraints',
        'Sub-questions:': 'sub_questions',
        'Answer format:': 'answer_format'
    }
    
    missing = []
    for field_marker, field_name in required_fields.items():
        if field_marker not in plan_content:
            missing.append(field_name)
    
    results['missing_fields'] = missing
    results['plan_complete'] = len(missing) == 0
    
    return results

# ============ Validation Functions ============
def validate_trajectory(trajectory: str, ground_truth: List[str]) -> Dict[str, Any]:
    """Validate trajectory quality"""
    results = {
        'valid': True,
        'has_answer': False,
        'answer_correct': False,
        'has_search': False,
        'has_filter': False,
        'has_rank': False,
        'has_plan': False,
        'plan_valid': False,
        'extracted_answer': None,
        'errors': []
    }
    
    # Validate plan (REQUIRED)
    plan_validation = validate_plan(trajectory)
    results['has_plan'] = plan_validation['has_plan']
    results['plan_valid'] = plan_validation['plan_complete'] and plan_validation['plan_position_valid']
    
    if not plan_validation['has_plan']:
        results['valid'] = False
        results['errors'].append("Missing <plan> tag")
    elif not plan_validation['plan_position_valid']:
        results['valid'] = False
        results['errors'].append("Plan is not at the beginning")
    elif not plan_validation['plan_complete']:
        results['valid'] = False
        results['errors'].append(f"Incomplete plan, missing fields: {plan_validation['missing_fields']}")
  
    # Check if has answer
    answer_match = re.findall(r'<answer>(.*?)</answer>', trajectory, re.DOTALL)
    if answer_match:
        results['has_answer'] = True
        extracted = answer_match[-1].strip()
        results['extracted_answer'] = extracted
        
        # 归一化
        normalized_pred = enhanced_normalize(extracted)
        
        # 先尝试换行分隔
        if '\n' in extracted:
            pred_list = [enhanced_normalize(p.strip()) for p in extracted.split('\n') if p.strip()]
        # 如果没有换行，尝试逗号分隔
        elif ',' in extracted:
            pred_list = [enhanced_normalize(p.strip()) for p in extracted.split(',') if p.strip()]
        # 否则作为单个答案
        else:
            pred_list = [normalized_pred]
        
        # 过滤太短的片段（避免误匹配）
        pred_list = [p for p in pred_list if len(p) >= 3]
        
    
    
        # 检查是否匹配
        for gt in ground_truth:
            normalized_gt = enhanced_normalize(gt)

            # 方法1：精确匹配（最可靠）
            if normalized_gt in pred_list:
                results['answer_correct'] = True
                break
            if results['answer_correct']:
                break
    else:
        results['valid'] = False
        results['errors'].append("No <answer> tag found")
    
    # Check other tags
    results['has_search'] = bool(re.search(r'<search>', trajectory))
    results['has_filter'] = bool(re.search(r'<filter>', trajectory))
    results['has_rank'] = bool(re.search(r'<rank>', trajectory))
    
    # Check tag balance
    for tag in ['plan', 'think', 'search', 'information', 'filter', 'rank', 'answer']:
        open_count = len(re.findall(f'<{tag}>', trajectory))
        close_count = len(re.findall(f'</{tag}>', trajectory))
        if open_count != close_count:
            results['valid'] = False
            results['errors'].append(f"Unbalanced <{tag}> tags: {open_count} vs {close_count}")
    
    return results


# ============ Main Generation Function ============
def generate_sft_dataset(
    input_file: str,
    output_file: str,
    num_samples: int = 100,
    start_index: int = 0,
    model_api: str = "openai",
    retriever_url: str = "http://127.0.0.1:8000/retrieve",
    filter_correct_only: bool = True,
    **api_kwargs
):
    """Generate SFT dataset with trajectories"""
    
    print(f"📖 Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Loaded {len(data)} examples")
    print(f"   Will generate {num_samples} trajectories")
    print(f"   Using model API: {model_api}")
    print(f"   Retriever URL: {retriever_url}")
    
    sft_data = []
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'correct': 0,
        'incorrect': 0,
        'has_plan': 0,
        'plan_valid': 0,
        'missing_plan': 0
    }
    
    # Select samples
    end_index = start_index + num_samples
    samples = data[start_index:end_index]
    print(f"   Generating samples from index {start_index} to {end_index-1}")

    for idx, example in enumerate(tqdm(samples, desc="Generating trajectories")):
        stats['total'] += 1
        
        question = example['question']
        answers = example['answer']
        if isinstance(answers, str):
            answers = [answers]
        print(f"\n[{idx+1}/{len(samples)}] Q: {question[:80]}...")
        
        # Generate trajectory
        trajectory = generate_trajectory_interactive(
            question=question,
            retriever_url=retriever_url,
            topk=50,
            model_api=model_api,
            **api_kwargs
        )
        
        if not trajectory:
            stats['failed'] += 1
            print("  ❌ Generation failed")
            continue
        
        # Validate trajectory
        validation = validate_trajectory(trajectory, answers)
        
        # Update plan statistics
        if validation['has_plan']:
            stats['has_plan'] += 1
        else:
            stats['missing_plan'] += 1
        
        if validation['plan_valid']:
            stats['plan_valid'] += 1
        
        if validation['valid'] and validation['answer_correct']:
            stats['success'] += 1
            stats['correct'] += 1
            print(f"  ✅ Valid & Correct | Pred: {validation['extracted_answer']} | GT: {answers[0]}")
            
            # Add to SFT dataset
            sft_example = {
                "quid": example.get('quid'),
                "qtype": example.get('qtype'),
                "question": question,
                "trajectory": trajectory,
                "ground_truth": answers,
                "validation": validation
            }
            sft_data.append(sft_example)
            
        elif validation['valid'] and not validation['answer_correct']:
            stats['incorrect'] += 1
            print(f"  ⚠️  Valid but Wrong | Pred: {validation['extracted_answer']} | GT: {answers[0]}")
            
            if not filter_correct_only:
                sft_example = {
                    "quid": example.get('quid'),
                    "qtype": example.get('qtype'),
                    "question": question,
                    "trajectory": trajectory,
                    "ground_truth": answers,
                    "validation": validation
                }
                sft_data.append(sft_example)
        else:
            stats['failed'] += 1
            print(f"  ❌ Invalid | Errors: {validation['errors']}")
        
        # Save intermediate results every 10 examples
        if (idx + 1) % 100 == 0:
            temp_file = output_file.replace('.jsonl', f'_temp_{idx+1}.jsonl')
            with open(temp_file, 'w') as f:
                for item in sft_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"  💾 Saved intermediate results to {temp_file}")
        
        # Rate limiting
        time.sleep(1.0)
    
    # Save final dataset
    print(f"\n💾 Saving final dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save statistics
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Generation Summary:")
    print(f"   Total processed: {stats['total']}")
    print(f"   Successful: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"   Correct answers: {stats['correct']} ({stats['correct']/stats['total']*100:.1f}%)")
    print(f"   Incorrect answers: {stats['incorrect']}")
    print(f"   Failed: {stats['failed']}")
    print(f"")
    print(f"   Plan Statistics:")
    print(f"   - Has plan: {stats['has_plan']} ({stats['has_plan']/stats['total']*100:.1f}%)")
    print(f"   - Valid plan: {stats['plan_valid']} ({stats['plan_valid']/stats['total']*100:.1f}%)")
    print(f"   - Missing plan: {stats['missing_plan']} ({stats['missing_plan']/stats['total']*100:.1f}%)")
    print(f"")
    print(f"   Final dataset size: {len(sft_data)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SFT trajectories with model API and retriever')
    
    # Data arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file (e.g., train_after_first.json)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file for SFT data')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to generate')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start index for generating samples')
    # Retriever arguments
    parser.add_argument('--retriever_url', type=str, default='http://127.0.0.1:8000/retrieve',
                        help='Retriever service URL')
    
    # Model API arguments
    parser.add_argument('--model_api', type=str, default='openai', choices=['openai', 'local'],
                        help='Model API type')
    parser.add_argument('--api_key', type=str, default="",
                        help='API key for OpenAI')
    parser.add_argument('--base_url', type=str, default="",
                        help='Base URL for OpenAI-compatible API')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model name')
    parser.add_argument('--model_url', type=str, default='http://localhost:8001/v1/chat/completions',
                        help='Local model URL (for local API)')
    
    # Other arguments
    parser.add_argument('--filter_correct_only', action='store_true',
                        help='Only keep trajectories with correct answers')
    
    args = parser.parse_args()
    
    # Prepare API kwargs
    api_kwargs = {}
    if args.model_api == 'openai':
        api_kwargs = {
            'model': args.model,
            'api_key': args.api_key,
            'base_url': args.base_url
        }
    elif args.model_api == 'local':
        api_kwargs = {
            'model_url': args.model_url,
            'model_name': args.model
        }
    
    generate_sft_dataset(
        input_file=args.input,
        output_file=args.output,
        num_samples=args.num_samples,
        start_index=args.start_index,
        model_api=args.model_api,
        retriever_url=args.retriever_url,
        filter_correct_only=args.filter_correct_only,
        **api_kwargs
    )