#!/usr/bin/env python3
"""
单模型评估脚本
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import ModelEvaluator
from config.system_config import SystemConfig
from utils.helpers import set_seed


def load_model(model_path: str, config):
    """加载模型"""
    from models.defender import DefenderModel
    model = DefenderModel(model_path, config, "eval_model")
    return model.model, model.tokenizer


def main():
    parser = argparse.ArgumentParser(description="单模型评估")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--name", type=str, default="evaluated_model", help="模型名称")
    parser.add_argument("--output", type=str, default="./results/evaluation_reports", help="输出目录")
    args = parser.parse_args()
    
    set_seed(42)
    config = SystemConfig()
    
    print("=" * 60)
    print("单模型评估")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print()
    
    # 加载模型
    print("加载模型...")
    model, tokenizer = load_model(args.model, config)
    
    # 创建评估器
    evaluator = ModelEvaluator(model, tokenizer, save_dir=args.output)
    
    # 执行评估
    metrics = evaluator.evaluate_all(args.name)
    
    # 打印结果
    print()
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"安全性得分: {metrics.safety_score:.4f}")
    print(f"有害内容率: {metrics.harmful_content_rate:.4f}")
    print(f"越狱成功率: {metrics.jailbreak_success_rate:.4f}")
    print()
    print(f"有用性得分: {metrics.helpfulness_score:.4f}")
    print(f"回答完整性: {metrics.answer_completeness:.4f}")
    print()
    print(f"鲁棒性得分: {metrics.robustness_score:.4f}")
    print(f"攻击抵抗率: {metrics.attack_resistance_rate:.4f}")
    print(f"回答一致性: {metrics.consistency_score:.4f}")
    print()
    print(f"平均响应时间: {metrics.avg_response_time:.2f}秒")
    print(f"平均Token数: {metrics.avg_token_count:.1f}")
    print()
    print(f"综合得分: {metrics.overall_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
