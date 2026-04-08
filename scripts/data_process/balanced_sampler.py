import json
import random
import os
import argparse
from collections import defaultdict, Counter

def print_stats(data, title="数据集统计"):
    """打印数据集的分布情况"""
    total = len(data)
    if total == 0:
        print(f"⚠️ {title}: 数据为空")
        return

    level_counts = Counter([item.get('question_level', 'unknown') for item in data])
    type_counts = Counter([item.get('answer_type', 'unknown') for item in data])

    print(f"\n📊 {title} (Total: {total}):")
    
    print(f"--- Answer Type 分布 (目标: ~{100/len(type_counts):.1f}%) ---")
    for k, v in type_counts.most_common():
        print(f"{k:<35} : {v:<5} ({v/total*100:.2f}%)")
        
    print(f"--- Question Level 分布 (目标: ~{100/len(level_counts):.1f}%) ---")
    for k, v in level_counts.most_common():
        print(f"{k:<35} : {v:<5} ({v/total*100:.2f}%)")
    print("-" * 60)

def balanced_sampling(input_path, output_path, target_size=600, seed=42):
    random.seed(seed)
    
    print(f"📖 正在读取源文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        # 兼容 JSON 和 JSONL
        try:
            full_data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            full_data = [json.loads(line) for line in f if line.strip()]
            
    print(f"✅ 源数据共 {len(full_data)} 条")
    
    # 1. 数据分组：按 Answer Type 分组
    type_groups = defaultdict(list)
    for item in full_data:
        a_type = item.get('answer_type', 'unknown')
        type_groups[a_type].append(item)
    
    unique_types = list(type_groups.keys())
    num_types = len(unique_types)
    
    # 计算 Answer Type 的理想配额
    # 例如 600 / 9 = 66.6 -> 每个类型理想有 66 个
    base_quota_per_type = target_size // num_types
    print(f"🎯 采样目标: 总数 {target_size}, 共有 {num_types} 种 Answer Type, 每种理想配额: {base_quota_per_type}")

    selected_data = []
    
    # 2. 第一轮采样：尽力满足 Answer Type 均衡，同时兼顾 Level 均衡
    # 记录哪些类型没喂饱配额，需要后续补充
    
    for a_type in unique_types:
        items = type_groups[a_type]
        count = len(items)
        
        # 如果当前类型的样本数少于理想配额，全拿走 (我们不能无中生有)
        if count <= base_quota_per_type:
            selected_data.extend(items)
        else:
            # 如果样本充足，我们需要在这一层内部再做一次 Level 均衡
            # 按 Level 分组
            level_groups = defaultdict(list)
            for item in items:
                level_groups[item.get('question_level', 'unknown')].append(item)
            
            # 计算 Level 的理想配额 (例如 66 / 3 = 22)
            unique_levels = list(level_groups.keys())
            if not unique_levels: # 防御性编程
                sampled_subset = random.sample(items, base_quota_per_type)
                selected_data.extend(sampled_subset)
                continue

            quota_per_level = base_quota_per_type // len(unique_levels)
            
            type_subset = []
            
            # Level 均衡采样
            for level in unique_levels:
                l_items = level_groups[level]
                if len(l_items) <= quota_per_level:
                    type_subset.extend(l_items)
                else:
                    type_subset.extend(random.sample(l_items, quota_per_level))
            
            # 如果 Level 均衡后还没填满该 Type 的配额 (因为整除问题)，随机补齐
            if len(type_subset) < base_quota_per_type:
                remaining_quota = base_quota_per_type - len(type_subset)
                # 从还没选中的里面补
                pool = [x for x in items if x not in type_subset] # 这里的查找效率较低，但数据量小没关系
                if pool:
                    type_subset.extend(random.sample(pool, min(len(pool), remaining_quota)))
            
            selected_data.extend(type_subset)

    # 3. 填补空缺 (Fill the Gap)
    # 因为某些稀有类（如 relation_union...）样本极少，第一轮可能只拿了 11 个，远不到 66 个。
    # 这会导致 selected_data 总数小于 600。我们需要从剩下的数据池里随机抽取来填满 600。
    
    current_count = len(selected_data)
    shortage = target_size - current_count
    
    if shortage > 0:
        print(f"⚠️ 第一轮均衡采样后只有 {current_count} 条，还缺 {shortage} 条。将从剩余数据中随机补充...")
        
        # 构建已选数据的 ID 集合（假设有 quid，或者用对象引用）
        # 为了简单，我们重建一个剩余数据池
        # 注意：这里为了效率，可以使用 ID Set。如果没有 ID，这里假设 list item 是可哈希的或者用 index
        
        # 简单方法：将 selected_data 中的元素标记
        # 这里使用一种简单粗暴但有效的方法：
        # 从 full_data 中移除已经选中的，然后 sample
        
        # 为了去重，我们假设数据中有 'quid' 或 'id'，如果没有，就用 json 字符串做 key
        seen_ids = set()
        for item in selected_data:
            # 优先用 id，没有则用 question 做唯一标识
            uid = item.get('quid', item.get('id', item.get('question')))
            seen_ids.add(uid)
            
        remaining_pool = []
        for item in full_data:
            uid = item.get('quid', item.get('id', item.get('question')))
            if uid not in seen_ids:
                remaining_pool.append(item)
        
        if len(remaining_pool) >= shortage:
            fillers = random.sample(remaining_pool, shortage)
            selected_data.extend(fillers)
        else:
            print("⚠️ 警告：源数据总量不足，无法填满 600 条！")
            selected_data.extend(remaining_pool)

    # 4. 再次打乱顺序
    random.shuffle(selected_data)
    
    # 5. 保存
    print(f"💾 保存结果至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)
        
    # 6. 打印最终统计
    print_stats(selected_data, "最终采样数据统计")

if __name__ == "__main__":
    # 请修改这里的路径指向你的【全量】训练数据
    # 注意：不要指向那个已经不均匀的 600 条文件，要指向大的源文件
    INPUT_FILE = "/data/TimelineKGQA/unified_kg_cron_questions_train.json" 
    
    # 输出路径
    OUTPUT_FILE = "/data/TimelineKGQA/unified_kg_cron_questions_train_balanced_1500.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=INPUT_FILE)
    parser.add_argument('--output', type=str, default=OUTPUT_FILE)
    parser.add_argument('--num', type=int, default=1500)
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在，如果不存在则尝试使用用户提供的那个600文件作为输入（虽然效果不好，但至少能跑）
    if not os.path.exists(args.input):
        fallback = "/data/TimelineKGQA/unified_kg_cron_questions_train_600.json"
        if os.path.exists(fallback):
            print(f"⚠️ 未找到全量文件，回退使用: {fallback} (注意：这可能导致无法改善平衡性)")
            args.input = fallback
    
    balanced_sampling(args.input, args.output, args.num)