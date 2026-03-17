"""
法官模型 - 对防御者回答进行评分（集成三库）
"""
import torch
import re
from typing import List, Dict, Tuple, Optional, Any  # ✅ 添加 Any
from .base_model import BaseModel
from knowledge.truth_base import TruthBase
from knowledge.experience_base import ExperienceBase
from knowledge.creativity_base import CreativityBase
import logging

logger = logging.getLogger(__name__)


class JudgeModel(BaseModel):
    """法官模型类 - 集成三库评分"""
    
    def __init__(self, model_path: str, config: Any, model_id: str = "judge",
                 truth_base: Optional[TruthBase] = None,
                 experience_base: Optional[ExperienceBase] = None,
                 creativity_base: Optional[CreativityBase] = None):
        # ✅ 法官不需要LoRA，注释掉setup_lora
        super().__init__(model_path, config, model_id)
        # self.setup_lora()  # 法官使用全模型
        
        self.evaluation_history: List[Dict] = []
        
        # 集成三个知识库
        self.truth_base = truth_base if truth_base else TruthBase()
        self.experience_base = experience_base if experience_base else ExperienceBase()
        self.creativity_base = creativity_base if creativity_base else CreativityBase()
        
        # 知识库权重配置
        self.truth_weight = 0.4
        self.experience_weight = 0.4
        self.creativity_weight = 0.2
