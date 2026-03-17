# 对抗性多智能体AI安全训练框架

## 📖 项目简介

这是一个基于**QLoRA微调**的对抗性多智能体AI安全训练框架，通过攻击者-防御者对抗训练，提升模型的鲁棒性和安全性。

### 核心特性

- ✅ **QLoRA 4-bit量化** - 显存节省80%+
- ✅ **多样性惩罚机制** - 防止策略过早收敛
- ✅ **动态及格线** - 课程学习思想
- ✅ **导师任期限制** - 防止局部最优锁定
- ✅ **进化机制** - 淘汰/杂交/突变
- ✅ **三库系统** - 真理库/经验库/奇思妙想库
- ✅ **RAG检索** - 带代价优化的外部知识检索

### 模型配置

| 角色 | 基础模型 | HuggingFace路径 |
|------|----------|-----------------|
| 攻击者 | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` |
| 防御者 | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` |
| 法官 | Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` |

## 🚀 快速开始

### 环境安装

```bash
# 创建虚拟环境
conda create -n adversarial-safety python=3.10
conda activate adversarial-safety

# 安装依赖
pip install -r requirements.txt
