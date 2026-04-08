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

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    print(golden_answers)
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def _get_assistant_content(text):
    """统一提取 assistant 回复内容的辅助函数"""
    # 1. 尝试匹配 Qwen 格式
    qwen_pattern = r"<\|im_start\|>assistant\s*"
    match = re.search(qwen_pattern, text)
    if match:
        return text[match.end():]

    # 2. 尝试匹配 LLaMA 3 格式 (允许换行符)
    llama_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*"
    match = re.search(llama_pattern, text)
    if match:
        return text[match.end():]
    
    # 3. 如果找不到 header，假设整段都是回复（兼容纯生成模式）
    return text

def is_valid_sequence(text):
    content = _get_assistant_content(text)
    
    # Extract the content after the assistant marker
    # start_pos = assistant_match.end()
    # content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["plan", "think", "search", "information", "filter", "rank", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:plan|think|search|information|filter|rank|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> plan -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:plan|think|search|information|filter|rank|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<plan>" and state in ["start"]:
                state = "in_plan"
            elif part == "</plan>" and state == "in_plan":
                state = "after_plan"
            elif part == "<think>" and state in ["after_plan", "information","after_rank","after_filter","after_search"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state in ["after_plan", "after_think"]:
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<filter>" and state in ["after_think", "information"]:
                state = "in_filter"
            elif part == "</filter>" and state == "in_filter":
                state = "after_filter"
            # 新增：允许 rank 出现在 filter 之后
            elif part == "<rank>" and state in ["after_filter","after_think","information"]:
                state = "in_rank"
            elif part == "</rank>" and state == "in_rank":
                state = "after_rank"
            elif part == "<answer>" and state in ["after_think", "after_rank", "after_filter", "information"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            allowed_content_states = ["in_plan", "in_think", "in_search", "in_information", "in_filter", "in_rank", "in_answer"]
            if state not in allowed_content_states:
                # 允许少量的空白或换行，但不允许大量幻觉文本
                if len(part.strip()) > 5: 
                    return False, f"Text found outside tags in state {state}: {part[:20]}..."
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_solution(solution_str):
    content = _get_assistant_content(solution_str)
    # Find all answer tags in assistant's response
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, content, re.DOTALL)
    
    if not matches:
        return None
    # 额外保护：过滤掉过长的答案（可能是误匹配）
    valid_matches = [m for m in matches if len(m) < 200]  # 假设答案不超过200字符
    
    if not valid_matches:
        # 如果所有匹配都太长，返回最短的一个
        return min(matches, key=len).strip()
    # Return the last answer (in case of multiple answers)
    return matches[-1].strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    assistant_str = _get_assistant_content(solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"assistant string: {assistant_str}")
            
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                return score # 1
            else:
                return score - structure_format_score # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return final_format_score # 0.1
