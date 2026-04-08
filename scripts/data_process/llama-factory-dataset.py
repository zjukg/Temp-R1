import json
import re
import os

def convert_to_llama_factory_format(input_file, output_file):
    print(f"正在读取文件: {input_file} ...")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件未找到 - {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            # 尝试作为整个 JSON 列表读取
            data = json.load(f)
        except json.JSONDecodeError:
            # 如果失败，尝试作为 JSONL (每行一个 JSON) 读取
            f.seek(0)
            data = [json.loads(line) for line in f]

    formatted_data = []

    # ==================== SYSTEM PROMPT 定义开始 ====================
    # 注意：这里必须使用三个引号 """ 来定义多行字符串
    system_prompt = """You are a question-answering assistant with access to a knowledge base.

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

Now, please answer the following question.
"""
    # ==================== SYSTEM PROMPT 定义结束 ====================

    for item in data:
        trajectory = item.get("trajectory", "")
        question = item.get("question", "")
        
        # 1. 初始化对话，放入 System Prompt 和 用户问题
        conversations = [
            {
                "from": "system",
                "value": system_prompt
            },
            {
                "from": "human",
                "value": question
            }
        ]

        # 2. 使用正则表达式拆分 trajectory
        # 逻辑：以 <information>...</information> 代码块为界进行切分
        parts = re.split(r'(<information>.*?</information>)', trajectory, flags=re.DOTALL)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 3. 判断当前片段是 GPT 生成的还是环境返回的 Information
            if part.startswith("<information>"):
                # 这是检索结果 -> Role: observation (LLaMA-Factory 会自动 Mask)
                conversations.append({
                    "from": "observation",
                    "value": part
                })
            else:
                # 这是模型生成的 (plan, think, search, answer) -> Role: gpt
                conversations.append({
                    "from": "gpt",
                    "value": part
                })

        # 4. 构建最终样本
        formatted_data.append({
            "conversations": conversations,
            # 可以保留其他元数据方便后续 debug
            "tools": [],
            "id": str(item.get("quid", ""))
        })

    # 5. 保存结果
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成！共处理 {len(formatted_data)} 条数据。")
    print(f"输出文件: {output_file}")

# 使用示例
if __name__ == "__main__":
    # input_path = "/outputs/gpt4o-1000-cron-new.jsonl"
    # output_path = "/LLaMA-Factory/data/cron_sft_train.json"
    input_path = "/outputs/train_sft_oversampled_merged.jsonl"
    output_path = "/LLaMA-Factory/data/multitq_sft_train.json"
    convert_to_llama_factory_format(input_path, output_path)