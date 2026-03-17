"""
训练配置 - 定义训练相关参数
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # ========== 对抗训练配置 ==========
    NUM_ATTACKERS: int = 4  # 攻击者数量
    NUM_DEFENDERS: int = 4  # 防御者数量
    NUM_ROUNDS: int = 50  # 总训练轮数
    
    # ========== 多样性惩罚配置 ==========
    DIVERSITY_PENALTY_WEIGHT: float = 0.15
    DIVERSITY_DECAY_RATE: float = 0.98
    DIVERSITY_MIN_PENALTY: float = 0.05
    DIVERSITY_HISTORY_WINDOW: int = 50  # 历史窗口大小
    SIMILARITY_THRESHOLD: float = 0.7  # 相似度阈值
    
    # ========== 动态及格线配置 ==========
    INITIAL_THRESHOLD: float = 0.6
    MIN_THRESHOLD: float = 0.4
    MAX_THRESHOLD: float = 0.95
    THRESHOLD_ADJUSTMENT_RATE: float = 0.05
    THRESHOLD_WINDOW_SIZE: int = 10
    
    # ========== 导师配置 ==========
    MAX_MENTOR_TERMS: int = 3  # 最大任期
    MENTOR_EXAM_INTERVAL: int = 10  # 导师考试间隔
    MENTOR_EVALUATION_ROUNDS: int = 10  # 导师评估轮数
    
    # ========== 考试配置 ==========
    EXAM_INTERVAL: int = 5  # 考试间隔
    EXTRA_TRAINING_ROUNDS: int = 2  # 未及格额外训练轮数
    
    # ========== 进化配置 ==========
    ELIMINATION_INTERVAL: int = 20  # 淘汰间隔
    ELIMINATION_COUNT: int = 2  # 淘汰数量
    HYBRID_ALPHA: float = 0.5  # 杂交混合比例
    MUTATION_NOISE_SCALE: float = 0.02  # 突变噪声
    
    # ========== RAG配置 ==========
    RAG_COST_COEFFICIENT: float = 0.2
    RAG_COST_DECAY_RATE: float = 0.8  # 停滞时降低代价
    STAGNATION_WINDOW: int = 5  # 停滞检测窗口
    
    # ========== DRPO配置 ==========
    DRPO_WEIGHT: float = 0.4
    JUDGE_WEIGHT: float = 0.4
    RAG_WEIGHT: float = 0.2
    
    def validate(self) -> bool:
        """验证配置"""
        checks = [
            0 < self.DIVERSITY_PENALTY_WEIGHT < 1,
            0 < self.DIVERSITY_DECAY_RATE <= 1,
            0 < self.INITIAL_THRESHOLD < 1,
            self.MIN_THRESHOLD < self.MAX_THRESHOLD,
            self.MAX_MENTOR_TERMS > 0,
            self.NUM_ATTACKERS > 0,
            self.NUM_DEFENDERS > 0,
        ]
        return all(checks)
