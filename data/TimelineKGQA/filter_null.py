import json
import argparse
import os

def filter_null_answers(input_json_path, output_json_path):
    # 读取原始数据
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"原始条目数: {len(data)}")

    # 过滤 answer 为 null 的条目
    filtered_data = [item for item in data if item.get("answer") is not None]

    print(f"过滤后条目数: {len(filtered_data)}")

    # 保存到新的 JSON 文件
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"过滤后的数据已保存到 {output_json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default="/disk1/gongzhaoyan/Second/Search-R1/data/TimelineKGQA/test_icews_actor_questions.json",
        type=str,
        help="输入 JSON 文件路径"
    )
    parser.add_argument(
        "--output_json",
        default="/disk1/gongzhaoyan/Second/Search-R1/data/TimelineKGQA/test_icews_actor_questions_filt.json",
        type=str,
        help="输出 JSON 文件路径"
    )
    args = parser.parse_args()

    filter_null_answers(args.input_json, args.output_json)
