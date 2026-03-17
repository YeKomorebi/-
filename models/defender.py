"""
防御者模型 - 回答攻击问题
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any  # ✅ 添加 Any
from .base_model import BaseModel
from training.drpo import DRPO
import logging

logger = logging.getLogger(__name__)


class DefenderModel(BaseModel):
    """防御者模型类"""
    
    def __init__(self, model_path: str, config: Any, model_id: str):
        super().__init__(model_path, config, model_id)
        self.setup_lora()
        
        # DRPO配置
        self.drpo = DRPO(
            drpo_weight=config.training.DRPO_WEIGHT,
            judge_weight=config.training.JUDGE_WEIGHT,
            rag_weight=config.training.RAG_WEIGHT,
        )
        
        # 防御统计
        self.defense_history: List[Dict] = []
        self.average_score: float = 0.0
        self.total_defenses: int = 0
        self.total_score: float = 0.0
        
        # RAG相关
        self.rag_usage_count: int = 0
        
        # ✅ 添加额外训练轮数属性
        self.extra_training_rounds: int = 0
