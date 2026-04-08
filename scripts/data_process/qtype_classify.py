#!/usr/bin/env python3
"""
Split MultiTQ training data by qtype
将MultiTQ训练集按qtype分类保存
"""

import json
from collections import defaultdict
from pathlib import Path
import argparse


def split_by_qtype(input_file, output_dir):
    """
    按qtype字段分割训练数据
    
    Args:
        input_file: 输入的train_MULTITQ.json文件路径
        output_dir: 输出目录
    """
    print(f"Loading data from {input_file}...")
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # 按qtype分组
    qtype_groups = defaultdict(list)
    qtype_counts = defaultdict(int)
    
    for example in data:
        qtype = example.get('qtype', 'unknown')
        qtype_groups[qtype].append(example)
        qtype_counts[qtype] += 1
    
    # 打印统计信息
    print(f"\nFound {len(qtype_groups)} different qtypes:")
    for qtype in sorted(qtype_counts.keys()):
        print(f"  - {qtype}: {qtype_counts[qtype]} examples")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存每个qtype为单独的文件
    print(f"\nSaving to {output_dir}...")
    for qtype, examples in qtype_groups.items():
        output_file = output_path / f"train_{qtype}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved {len(examples)} examples to {output_file}")
    
    # 保存一个统计文件
    stats = {
        'total_examples': len(data),
        'num_qtypes': len(qtype_groups),
        'qtype_distribution': dict(qtype_counts)
    }
    
    stats_file = output_path / "qtype_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved statistics to {stats_file}")
    
    print(f"\n✅ Done! Split {len(data)} examples into {len(qtype_groups)} qtype groups.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split MultiTQ training data by qtype')
    parser.add_argument(
        '--input',
        type=str,
        default='/data/MultiTQ/train_MULTITQ.json',
        help='Input training data file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/MultiTQ/split_by_qtype',
        help='Output directory for split files'
    )
    
    args = parser.parse_args()
    
    split_by_qtype(args.input, args.output_dir)