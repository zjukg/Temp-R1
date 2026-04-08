#!/usr/bin/env python3

import json
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import time
import re
import concurrent.futures
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 全局锁，用于文件写入
FILE_LOCK = threading.Lock()

# 全局 Session 对象，用于复用 TCP 连接
RETRIEVER_SESSION = requests.Session()
MODEL_SESSION = requests.Session()

# 配置 Session 重试策略
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
RETRIEVER_SESSION.mount('http://', HTTPAdapter(max_retries=retries))
MODEL_SESSION.mount('https://', HTTPAdapter(max_retries=retries))

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

# ============ Retriever Functions ============
def query_retriever(question: str, retriever_url: str = "http://127.0.0.1:8000/retrieve", topk: int = 30) -> List[str]:
    """Query the retrieval service using global session"""
    try:
        payload = {
            "queries": [question],      
            "topk": topk,               
            "return_scores": True       
        }
        
        response = RETRIEVER_SESSION.post(
            retriever_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        results = response.json()
        
        formatted = []
        if 'result' in results and len(results['result']) > 0:
            retrieval_result = results['result'][0] 
            for idx, doc_item in enumerate(retrieval_result, 1):
                try:
                    content = doc_item['document']['contents']
                    if "\n" in content:
                        title = content.split("\n")[0]
                        text = "\n".join(content.split("\n")[1:])
                        formatted.append(f"Doc {idx}(Title: {title}) {text}")
                    else:
                        formatted.append(f"Doc {idx}: {content}")
                except (KeyError, TypeError):
                    if isinstance(doc_item, dict):
                        content = str(doc_item.get('document', doc_item))
                        formatted.append(f"Doc {idx}: {content}")
        return formatted
    
    except Exception as e:
        return []

def format_information(facts: List[str]) -> str:
    return "\n".join(facts)

# ============ Model API Functions (FIXED) ============
# ✅ 修复点：添加 **kwargs 来接收多余的参数，防止 TypeError
def call_openai_api(prompt: str, model: str = "gpt-4o", api_key: str = None, base_url: str = None, max_retries: int = 3, **kwargs) -> str:
    """Call OpenAI API using requests session"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4096
    }

    url = f"{base_url}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"

    for attempt in range(max_retries):
        try:
            response = MODEL_SESSION.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None
    return None

# ✅ 修复点：添加 **kwargs
def call_local_vllm(prompt: str, model_url: str, model_name: str, **kwargs) -> str:
    try:
        response = MODEL_SESSION.post(
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
    except Exception:
        return None

def truncate_at_first_action(response: str) -> str:
    search_pos = response.find('</search>')
    answer_pos = response.find('</answer>')
    positions = [(pos, tag) for pos, tag in [(search_pos, '</search>'), (answer_pos, '</answer>')] if pos != -1]
    
    if not positions:
        return response
    
    first_pos, first_tag = min(positions, key=lambda x: x[0])
    return response[:first_pos + len(first_tag)]

def extract_search_query(response: str) -> str:
    match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
    return match.group(1).strip() if match else ""

# ============ Interactive Trajectory Generation ============
def generate_trajectory_interactive(
    question: str,
    retriever_url: str,
    topk: int,
    model_api: str,
    **api_kwargs
) -> str:
    full_trajectory = ""
    conversation_history = INSTRUCTION.replace("{question}", question)
    max_turns = 4 
    
    for turn in range(max_turns):
        if model_api == "openai":
            # ✅ api_kwargs 里包含所有参数，call_openai_api 会忽略多余的
            response = call_openai_api(conversation_history, **api_kwargs)
        elif model_api == "local":
            response = call_local_vllm(conversation_history, **api_kwargs)
        else:
            return None
        
        if not response:
            break
        
        response = truncate_at_first_action(response)
        full_trajectory += response
        
        if '</search>' in response:
            search_query = extract_search_query(response)
            if not search_query:
                break
            
            retrieved_facts = query_retriever(search_query, retriever_url, topk)
            
            if not retrieved_facts:
                information_block = "\n\n<information>\nNo relevant information found.\n</information>\n\n"
            else:
                formatted_info = format_information(retrieved_facts)
                information_block = f"\n\n<information>\n{formatted_info}\n</information>\n\n"
            
            full_trajectory += information_block
            conversation_history = INSTRUCTION.replace("{question}", question) + "\n\n" + full_trajectory
            
        elif '</answer>' in response:
            break
            
        elif any(tag in response for tag in ['</think>', '</plan>', '</filter>', '</rank>']):
            conversation_history = INSTRUCTION.replace("{question}", question) + "\n\n" + full_trajectory
            
        else:
            if turn > 0: break
            conversation_history = INSTRUCTION.replace("{question}", question) + "\n\n" + full_trajectory
    
    return full_trajectory

def enhanced_normalize(text: str) -> str:
    text = text.replace(' to ', ' - ') 
    text = text.replace(' and ', ' - ') 
    
    date_range_match = re.match(r'(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})', text.strip())
    if date_range_match:
        d1, d2 = date_range_match.groups()
        return f"{min(d1, d2)} - {max(d1, d2)}"
        
    format_chars = ['_', '(', ')', '/', ',', ';', ':', '|']
    for char in format_chars:
        text = text.replace(char, ' ')
        
    from verl.utils.reward_score.qa_em_format import normalize_answer
    text = normalize_answer(text)
    text = ' '.join(text.split())
    return text

# ============ Validation Functions ============
def validate_plan(trajectory: str) -> Dict[str, Any]:
    results = {'has_plan': False, 'plan_complete': False, 'plan_content': None}
    plan_match = re.search(r'<plan>(.*?)</plan>', trajectory, re.DOTALL)
    if not plan_match: return results
    results['has_plan'] = True
    plan_content = plan_match.group(1).strip()
    results['plan_content'] = plan_content
    required_fields = ['Question type:', 'Time constraints:', 'Sub-questions:', 'Answer format:']
    if all(field in plan_content for field in required_fields):
        results['plan_complete'] = True
    return results

def validate_trajectory(trajectory: str, ground_truth: List[str]) -> Dict[str, Any]:
    results = {
        'valid': True, 'has_answer': False, 'answer_correct': False,
        'extracted_answer': None, 'errors': []
    }
    
    if not validate_plan(trajectory)['has_plan']:
        results['valid'] = False
        results['errors'].append("Missing <plan>")

    answer_match = re.findall(r'<answer>(.*?)</answer>', trajectory, re.DOTALL)
    if answer_match:
        results['has_answer'] = True
        extracted = answer_match[-1].strip()
        results['extracted_answer'] = extracted
        normalized_pred = enhanced_normalize(extracted)
        pred_list = [normalized_pred]
        if '\n' in extracted:
            pred_list.extend([enhanced_normalize(p.strip()) for p in extracted.split('\n') if p.strip()])
        if ',' in extracted:
            pred_list.extend([enhanced_normalize(p.strip()) for p in extracted.split(',') if p.strip()])
        
        for gt in ground_truth:
            normalized_gt = enhanced_normalize(gt)
            if normalized_gt in pred_list:
                results['answer_correct'] = True
                break
            if "no answer" in normalized_gt.lower() and "no answer" in normalized_pred.lower():
                results['answer_correct'] = True
                break
    else:
        results['valid'] = False
        results['errors'].append("No <answer>")
    return results

# ============ Reverse Prompt Logic ============
def get_reverse_generation_prompt(example: Dict[str, Any]) -> str:
    question = example['question']
    gt_list = example['answer']
    if isinstance(gt_list, str): gt_list = [gt_list]
    target_answer = str(gt_list[0]).strip()
    
    ans_lower = target_answer.lower()
    is_no_answer = "no answer" in ans_lower
    date_range_pattern = r'\d{4}-\d{2}-\d{2}.*?\d{4}-\d{2}-\d{2}'
    is_date_range = bool(re.search(date_range_pattern, target_answer))
    is_duration = any(k in ans_lower for k in ['year', 'month', 'day']) and not is_date_range
    is_boolean = ans_lower in ['longer', 'shorter', 'earlier', 'later', 'yes', 'no']
    is_digit = target_answer.isdigit()

    prompt_suffix = f"\n\n=========================================\n" \
                    f"[DATA GENERATION MODE: REVERSE REASONING]\n" \
                    f"The TARGET ANSWER for this question is: \"{target_answer}\"\n" \
                    f"Your task is to generate a valid reasoning trajectory that leads EXACTLY to this answer.\n" \
                    f"⚠️ CRITICAL INSTRUCTION: Output the answer string VERBATIM (word-for-word).\n" 

    if is_no_answer:
        prompt_suffix += "Special Instruction for 'No Answer': In <filter>, show retrieved docs DON'T match. Output <answer>No Answer</answer>."
    elif is_digit:
        prompt_suffix += f"Special Instruction for Integer: Dataset requires numeric answer. DO NOT output entity names. Explain counting/ranking logic in <think>. Output exactly: <answer>{target_answer}</answer>"
    elif is_date_range:
        prompt_suffix += f"Special Instruction for Time Range: This is a TIME PERIOD. DO NOT calc duration. Output exactly: <answer>{target_answer}</answer>"
    elif is_duration:
        prompt_suffix += f"Special Instruction for Duration: State 'Calculating duration...' in <think>. DO NOT recalc manually. TRUST target answer. Output exactly: <answer>{target_answer}</answer>"
    elif is_boolean:
        prompt_suffix += f"Special Instruction for Comparison: Compare values in <think>. Conclude with: <answer>{target_answer}</answer>"
    else:
        prompt_suffix += f"Instruction: Use <filter> and <rank> to narrow down. Output exactly: <answer>{target_answer}</answer>"
    
    prompt_suffix += "\n========================================="
    return question + prompt_suffix

def is_hard_case(example: Dict[str, Any]) -> bool:
    gt_list = example['answer']
    if isinstance(gt_list, str): gt_list = [gt_list]
    ans_str = str(gt_list[0]).strip().lower()
    if ans_str.isdigit(): return True
    if "no answer" in ans_str: return True
    if 'year' in ans_str or 'month' in ans_str or 'day' in ans_str:
        if len(ans_str) > 5: return True
    if ('-' in ans_str or ',' in ans_str or 'and' in ans_str) and len(ans_str) > 15 and any(c.isdigit() for c in ans_str): return True
    if ans_str in ['longer', 'shorter', 'earlier', 'later', 'before', 'after', 'yes', 'no']: return True
    return False

# ============ Worker Function ============
def process_single_example(example: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to process a single example"""
    original_question = example['question']
    answers = example['answer']
    if isinstance(answers, str): answers = [answers]
    
    if is_hard_case(example):
        prompt_question = get_reverse_generation_prompt(example)
    else:
        prompt_question = original_question
        
    trajectory = generate_trajectory_interactive(
        question=prompt_question,
        retriever_url=config['retriever_url'],
        topk=50,
        model_api=config['model_api'],
        **config['api_kwargs'] # 现在可以安全地解包所有参数了
    )
    
    if not trajectory:
        return None
        
    validation = validate_trajectory(trajectory, answers)
    
    result = {
        "quid": example.get('quid'),
        "qtype": example.get('qtype'),
        "question": original_question,
        "trajectory": trajectory,
        "ground_truth": answers,
        "validation": validation,
        "is_hard_case": is_hard_case(example)
    }
    
    return result

# ============ Main Generation Function ============
def generate_sft_dataset(
    input_file: str,
    output_file: str,
    num_samples: int = 100,
    start_index: int = 0,
    max_workers: int = 10,
    **kwargs
):
    print(f"📖 Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    end_index = start_index + num_samples
    samples = data[start_index:end_index]
    print(f"🚀 Starting generation with {max_workers} threads...")
    
    config = {
        'retriever_url': kwargs.get('retriever_url'),
        'model_api': kwargs.get('model_api'),
        'api_kwargs': {
            'model': kwargs.get('model'),
            'api_key': kwargs.get('api_key'),
            'base_url': kwargs.get('base_url'),
            'model_url': kwargs.get('model_url'),
            'model_name': kwargs.get('model_name')
        },
        'filter_correct_only': kwargs.get('filter_correct_only', True)
    }

    stats = {'total': 0, 'success': 0, 'correct': 0}
    
    with open(output_file, 'w') as f:
        pass 

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_single_example, ex, config): idx for idx, ex in enumerate(samples)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(samples), desc="Generating"):
            stats['total'] += 1
            idx = future_to_idx[future]
            
            try:
                result = future.result()
                if result is None:
                    continue
                    
                validation = result['validation']
                is_correct = validation['valid'] and validation['answer_correct']
                is_valid_wrong = validation['valid'] and not validation['answer_correct']
                
                should_save = False
                if is_correct:
                    stats['success'] += 1
                    stats['correct'] += 1
                    should_save = True
                elif is_valid_wrong and not config['filter_correct_only']:
                    should_save = True
                
                if should_save:
                    with FILE_LOCK:
                        with open(output_file, 'a') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            
            except Exception as exc:
                print(f"⚠️ Sample {idx} generated an exception: {exc}")

    print(f"\n✅ Generation complete.")
    print(f"   Total processed: {stats['total']}")
    print(f"   Success (Correct): {stats['correct']}")
    print(f"   Result saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--retriever_url', type=str, default='http://127.0.0.1:8000/retrieve')
    parser.add_argument('--model_api', type=str, default='openai', choices=['openai', 'local'])
    parser.add_argument('--api_key', type=str, default="")
    parser.add_argument('--base_url', type=str, default="")
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--filter_correct_only', action='store_true')
    
    args = parser.parse_args()
    
    generate_sft_dataset(
        input_file=args.input,
        output_file=args.output,
        num_samples=args.num_samples,
        start_index=args.start_index,
        max_workers=args.max_workers,
        retriever_url=args.retriever_url,
        model_api=args.model_api,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        filter_correct_only=args.filter_correct_only
    )