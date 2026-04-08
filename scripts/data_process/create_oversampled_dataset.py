#!/usr/bin/env python3
"""
根据题型预期准确率创建过采样数据集
确保过滤后每种题型至少有100条正确数据
"""

import json
import random
from pathlib import Path

# ============ 配置 ============
DATA_DIR = Path("data/MultiTQ/split_by_qtype")
OUTPUT_FILE = "data/MultiTQ/oversampled_for_sft.json"
TARGET_CORRECT_PER_TYPE = 100  # 目标：每种100条正确
RANDOM_SEED = 42

# 题型及其预期准确率（根据实际测试调整）
QTYPE_CONFIG = {
    "equal": {
        "expected_accuracy": 0.60,  # 60%准确率
        "sample_size": int(100 / 0.60 * 1.3)  # 217条（1.3倍安全系数）
    },
    "before_last": {
        "expected_accuracy": 0.30,
        "sample_size": int(100 / 0.30 * 1.3)  # 325条
    },
    "before_after": {
        "expected_accuracy": 0.70,
        "sample_size": int(100 / 0.70 * 1.3)  # 260条
    },
    "first_last": {
        "expected_accuracy": 0.70,
        "sample_size": int(100 / 0.70 * 1.3)  # 260条
    },
    "after_first": {
        "expected_accuracy": 0.30,  # 您提到的30%，保守估计35%
        "sample_size": int(100 / 0.30 * 1.3)  # 371条
    },
    "equal_multi": {
        "expected_accuracy": 0.40,  # 多答案题难度高
        "sample_size": int(100 / 0.30 * 1.3)  # 371条
    }
}

def load_and_sample(qtype: str, n_samples: int) -> list:
    """从指定qtype文件中随机采样"""
    file_path = DATA_DIR / f"train_{qtype}.json"
    
    print(f"📖 Loading {qtype}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"   Total available: {len(data):,} examples")
    
    if len(data) < n_samples:
        print(f"   ⚠️  Only {len(data):,} available (need {n_samples})")
        print(f"   → Using all {len(data):,} examples")
        sampled = data
    else:
        sampled = random.sample(data, n_samples)
        print(f"   ✅ Sampled {n_samples:,} examples")
    
    return sampled

def main():
    random.seed(RANDOM_SEED)
    
    print("="*70)
    print("🎯 Creating Oversampled SFT Dataset")
    print("="*70)
    print(f"Target: {TARGET_CORRECT_PER_TYPE} correct samples per type")
    print(f"Strategy: Oversample based on expected accuracy\n")
    
    all_samples = []
    total_samples = 0
    
    print("📊 Sampling Plan:")
    print(f"{'Type':<15} {'Accuracy':<10} {'Sample Size':<12} {'Expected Correct'}")
    print("-" * 70)
    
    for qtype, config in QTYPE_CONFIG.items():
        acc = config['expected_accuracy']
        size = config['sample_size']
        expected = int(size * acc)
        print(f"{qtype:<15} {acc*100:>6.0f}%     {size:>6,}       ~{expected}")
        total_samples += size
    
    print("-" * 70)
    print(f"{'TOTAL':<15} {'':10} {total_samples:>6,}       ~{TARGET_CORRECT_PER_TYPE * 6}")
    print()
    
    # 开始采样
    stats = {}
    for qtype, config in QTYPE_CONFIG.items():
        samples = load_and_sample(qtype, config['sample_size'])
        all_samples.extend(samples)
        stats[qtype] = {
            'sampled': len(samples),
            'expected_accuracy': config['expected_accuracy'],
            'expected_correct': int(len(samples) * config['expected_accuracy'])
        }
    
    # 打乱顺序
    print(f"\n🔀 Shuffling {len(all_samples):,} samples...")
    random.shuffle(all_samples)
    
    # 保存
    print(f"💾 Saving to {OUTPUT_FILE}...")
    OUTPUT_FILE_PATH = Path(OUTPUT_FILE)
    OUTPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats_file = OUTPUT_FILE_PATH.with_suffix('.stats.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'total_samples': len(all_samples),
            'target_correct_per_type': TARGET_CORRECT_PER_TYPE,
            'qtypes': stats
        }, f, indent=2)
    
    # 总结
    print(f"\n{'='*70}")
    print(f"✅ Oversampled Dataset Created")
    print(f"{'='*70}")
    print(f"   Total samples: {len(all_samples):,}")
    print(f"   Expected correct: ~{TARGET_CORRECT_PER_TYPE * 6} (if accuracy estimates hold)")
    print(f"\n   Per-type breakdown:")
    for qtype, stat in stats.items():
        print(f"      {qtype:<15}: {stat['sampled']:>4} samples "
              f"(~{stat['expected_correct']} correct at {stat['expected_accuracy']*100:.0f}%)")
    print(f"\n   Saved to: {OUTPUT_FILE}")
    print(f"   Stats: {stats_file}")
    print(f"{'='*70}")
    print(f"\n📝 Next step:")
    print(f"   bash scripts/data_process/generate_sft_tra.sh")
    print(f"   (will generate trajectories for all {len(all_samples):,} samples)")

if __name__ == "__main__":
    main()