"""
攻击者模型 - 生成攻击问题
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Any  # ✅ 添加 Any
from .base_model import BaseModel
from training.diversity_penalty import DiversityPenalty
import logging

logger = logging.getLogger(__name__)


class AttackerModel(BaseModel):
    """攻击者模型类"""
    
    def __init__(self, model_path: str, config: Any, model_id: str):
        super().__init__(model_path, config, model_id)
        self.setup_lora()
        
        # 多样性惩罚
        self.diversity_penalty = DiversityPenalty(
            penalty_weight=config.training.DIVERSITY_PENALTY_WEIGHT,
            decay_rate=config.training.DIVERSITY_DECAY_RATE,
            min_penalty=config.training.DIVERSITY_MIN_PENALTY,
            history_window=config.training.DIVERSITY_HISTORY_WINDOW,
        )
        
        # 攻击统计
        self.attack_history: List[Dict] = []
        self.success_rate: float = 0.0
        self.total_attacks: int = 0
        self.successful_attacks: int = 0
        
        # ✅ 添加额外训练轮数属性
        self.extra_training_rounds: int = 0
