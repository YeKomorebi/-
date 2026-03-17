#!/usr/bin/env python3
"""
模型对比脚本 - 比较原始模型与对抗训练后模型
"""
import argparse
import sys
from pathlib import Path
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import ModelEvaluator
from evaluation.metrics import EvaluationMetrics
from utils.helpers import set_seed


def load_model(model_path: str, config):
    """加载模型"""
    from models.defender import DefenderModel
    model = DefenderModel(model_path, config, "eval_model")
    return model.model, model.tokenizer


def compare_models(original_path: str, trained_path: str, config) -> Dict:
    """
    对比两个模型
    
    Args:
        original_path: 原始模型路径
        trained_path: 训练后模型路径
        config: 配置
        
    Returns:
        对比结果
    """
    set_seed(42)
    
    # 加载模型
    print("加载原始模型...")
    orig_model, orig_tokenizer = load_model(original_path, config)
    
    print("加载训练后模型...")
    trained_model, trained_tokenizer = load_model(trained_path, config)
    
    # 创建评估器
    print("创建评估器...")
    orig_evaluator = ModelEvaluator(orig_model, orig_tokenizer)
    trained_evaluator = ModelEvaluator(trained_model, trained_tokenizer)
    
    # 执行评估
    print("评估原始模型...")
    orig_metrics = orig_evaluator.evaluate_all("original_model")
    
    print("评估训练后模型...")
    trained_metrics = trained_evaluator.evaluate_all("trained_model")
    
    # 计算改进率
    trained_metrics.compute_improvement(orig_metrics)
    
    # 生成对比报告
    comparison = {
        "original": orig_metrics.to_dict(),
        "trained": trained_metrics.to_dict(),
        "improvement": {
            "overall": trained_metrics.improvement_rate,
            "safety": (trained_metrics.safety_score - orig_metrics.safety_score) / max(orig_metrics.safety_score, 0.001),
            "helpfulness": (trained_metrics.helpfulness_score - orig_metrics.helpfulness_score) / max(orig_metrics.helpfulness_score, 0.001),
            "robustness": (trained_metrics.robustness_score - orig_metrics.robustness_score) / max(orig_metrics.robustness_score, 0.001),
        }
    }
    
    return comparison


def generate_comparison_report(comparison: Dict, save_path: str = "./results/comparison_report.json"):
    """生成对比报告"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"对比报告已保存: {save_path}")


def generate_visualization(comparison: Dict, save_path: str = "./results/comparison_chart.png"):
    """生成可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 综合得分对比
    ax1 = axes[0, 0]
    categories = ['安全性', '有用性', '鲁棒性', '综合得分']
    orig_scores = [
        comparison['original']['safety']['safety_score'],
        comparison['original']['helpfulness']['helpfulness_score'],
        comparison['original']['robustness']['robustness_score'],
        comparison['original']['overall']['overall_score'],
    ]
    trained_scores = [
        comparison['trained']['safety']['safety_score'],
        comparison['trained']['helpfulness']['helpfulness_score'],
        comparison['trained']['robustness']['robustness_score'],
        comparison['trained']['overall']['overall_score'],
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, orig_scores, width, label='原始模型', color='#FF6B6B')
    ax1.bar(x + width/2, trained_scores, width, label='训练后模型', color='#4ECDC4')
    ax1.set_ylabel('得分')
    ax1.set_title('模型性能对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=15)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 改进率
    ax2 = axes[0, 1]
    improvement_cats = ['安全性', '有用性', '鲁棒性', '综合']
    improvements = [
        comparison['improvement']['safety'] * 100,
        comparison['improvement']['helpfulness'] * 100,
        comparison['improvement']['robustness'] * 100,
        comparison['improvement']['overall'] * 100,
    ]
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    ax2.bar(improvement_cats, improvements, color=colors)
    ax2.set_ylabel('改进率 (%)')
    ax2.set_title('性能改进率')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 安全性详细指标
    ax3 = axes[1, 0]
    safety_cats = ['安全得分', '有害内容率', '越狱成功率']
    orig_safety = [
        comparison['original']['safety']['safety_score'],
        comparison['original']['safety']['harmful_content_rate'],
        comparison['original']['safety']['jailbreak_success_rate'],
    ]
    trained_safety = [
        comparison['trained']['safety']['safety_score'],
        comparison['trained']['safety']['harmful_content_rate'],
        comparison['trained']['safety']['jailbreak_success_rate'],
    ]
    
    x = np.arange(len(safety_cats))
    ax3.bar(x - width/2, orig_safety, width, label='原始模型', color='#FF6B6B')
    ax3.bar(x + width/2, trained_safety, width, label='训练后模型', color='#4ECDC4')
    ax3.set_ylabel('得分/比率')
    ax3.set_title('安全性指标对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(safety_cats, rotation=15)
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 雷达图
    ax4 = axes[1, 1]
    radar_cats = ['安全性', '有用性', '鲁棒性', '效率']
    orig_radar = [
        comparison['original']['safety']['safety_score'],
        comparison['original']['helpfulness']['helpfulness_score'],
        comparison['original']['robustness']['robustness_score'],
        1 - comparison['original']['efficiency']['avg_response_time'] / 5,
    ]
    trained_radar = [
        comparison['trained']['safety']['safety_score'],
        comparison['trained']['helpfulness']['helpfulness_score'],
        comparison['trained']['robustness']['robustness_score'],
        1 - comparison['trained']['efficiency']['avg_response_time'] / 5,
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(radar_cats), endpoint=False).tolist()
    orig_radar += orig_radar[:1]
    trained_radar += trained_radar[:1]
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, polar=True)
    ax4.plot(angles, orig_radar, 'o-', linewidth=2, label='原始模型', color='#FF6B6B')
    ax4.plot(angles, trained_radar, 'o-', linewidth=2, label='训练后模型', color='#4ECDC4')
    ax4.fill(angles, orig_radar, alpha=0.25, color='#FF6B6B')
    ax4.fill(angles, trained_radar, alpha=0.25, color='#4ECDC4')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_cats)
    ax4.set_ylim(0, 1)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax4.set_title('性能雷达图')
    ax4.grid(True)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存: {save_path}")


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="模型对比评估")
    parser.add_argument("--original", type=str, required=True, help="原始模型路径")
    parser.add_argument("--trained", type=str, required=True, help="训练后模型路径")
    parser.add_argument("--output", type=str, default="./results", help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    
    from config.system_config import SystemConfig
    config = SystemConfig()
    
    print("=" * 60)
    print("模型对比评估")
    print("=" * 60)
    print(f"原始模型: {args.original}")
    print(f"训练后模型: {args.trained}")
    print()
    
    # 执行对比
    comparison = compare_models(args.original, args.trained, config)
    
    # 生成报告
    report_path = Path(args.output) / "comparison_report.json"
    generate_comparison_report(comparison, str(report_path))
    
    # 生成可视化
    chart_path = Path(args.output) / "comparison_chart.png"
    generate_visualization(comparison, str(chart_path))
    
    # 打印摘要
    print()
    print("=" * 60)
    print("评估摘要")
    print("=" * 60)
    print(f"原始模型综合得分: {comparison['original']['overall']['overall_score']:.4f}")
    print(f"训练后模型综合得分: {comparison['trained']['overall']['overall_score']:.4f}")
    print(f"综合改进率: {comparison['improvement']['overall']*100:.2f}%")
    print()
    print("各维度改进:")
    print(f"  安全性: {comparison['improvement']['safety']*100:+.2f}%")
    print(f"  有用性: {comparison['improvement']['helpfulness']*100:+.2f}%")
    print(f"  鲁棒性: {comparison['improvement']['robustness']*100:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
