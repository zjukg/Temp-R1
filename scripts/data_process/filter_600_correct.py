#!/usr/bin/env python3
"""
从生成的数据中筛选600条正确的（每种题型100条）
"""

import json
from pathlib import Path
from collections import defaultdict
import random

INPUT_FILE = "data/multitq_search/oversampled_sft.jsonl"
OUTPUT_FILE = "data/multitq_search/balanced_sft_600.jsonl"
TARGET_PER_TYPE = 100

def main():
    print("📊 Filtering correct samples...")
    
    # 按qtype分组
    data_by_qtype = defaultdict(list)
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            # 只保留valid且correct的
            if (item.get('validation', {}).get('valid', False) and 
                item.get('validation', {}).get('answer_correct', False)):
                
                qtype = item.get('qtype', 'unknown')
                data_by_qtype[qtype].append(item)
    
    # 统计
    print("\n📈 Correct samples by qtype:")
    for qtype in sorted(data_by_qtype.keys()):
        count = len(data_by_qtype[qtype])
        status = "✅" if count >= TARGET_PER_TYPE else "⚠️"
        print(f"   {status} {qtype:<15}: {count:>3} correct")
    
    # 选择每种100条
    final_data = []
    insufficient = []
    
    for qtype, samples in data_by_qtype.items():
        if len(samples) >= TARGET_PER_TYPE:
            # 随机选100条
            selected = random.sample(samples, TARGET_PER_TYPE)
        else:
            # 不够100条，全部使用
            selected = samples
            insufficient.append((qtype, len(samples)))
        
        final_data.extend(selected)
    
    # 打乱
    random.shuffle(final_data)
    
    # 保存
    print(f"\n💾 Saving {len(final_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 总结
    print(f"\n{'='*60}")
    print(f"✅ Filtered Dataset Created")
    print(f"{'='*60}")
    print(f"   Total samples: {len(final_data)}")
    
    if insufficient:
        print(f"\n   ⚠️  Insufficient qtypes (need to generate more):")
        for qtype, count in insufficient:
            print(f"      {qtype}: only {count}/{TARGET_PER_TYPE}")
        print(f"\n   💡 Suggestion: Increase sample size for these types")
    else:
        print(f"\n   ✅ All qtypes have {TARGET_PER_TYPE} samples!")
    
    print(f"\n   Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    random.seed(42)
    main()