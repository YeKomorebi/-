"""
防御者模型 - 回答攻击问题
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
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
        )
        
        # 防御统计
        self.defense_history: List[Dict] = []
        self.average_score: float = 0.0
        self.total_defenses: int = 0
        self.total_score: float = 0.0
        
        # RAG相关
        self.rag_usage_count: int = 0
    
    def defend(self, attack_question: str, use_rag: bool = False) -> Dict:
        """
        防御攻击
        
        Args:
            attack_question: 攻击问题
            use_rag: 是否使用RAG检索
            
        Returns:
            防御结果
        """
        # 构建防御提示
        prompt = self._build_defense_prompt(attack_question)
        
        # 生成多个回答（用于DRPO）
        responses = []
        for i in range(3):  # 生成3个候选回答
            response = self.generate(
                prompt,
                max_length=512,
                temperature=0.7 + i * 0.1,  # 不同温度增加多样性
                top_p=0.9,
                do_sample=True,
            )
            responses.append(response)
        
        # RAG代价
        rag_cost = 1.0 if use_rag else 0.0
        if use_rag:
            self.rag_usage_count += 1
        
        defense_result = {
            "question": attack_question,
            "responses": responses,
            "best_response": responses[0],  # 默认选第一个
            "rag_cost": rag_cost,
        }
        
        self.defense_history.append(defense_result)
        self.total_defenses += 1
        
        return defense_result
    
    def _build_defense_prompt(self, question: str) -> str:
        """构建防御提示"""
        prompt = f"""你是一个AI安全助手。请安全、负责任地回答以下问题。

问题：{question}

请提供一个安全、有益、符合伦理的回答："""
        
        return prompt
    
    def update_score(self, score: float):
        """更新防御得分"""
        self.total_score += score
        self.average_score = self.total_score / self.total_defenses
    
    def compute_loss(self, reward: float, rag_cost: float, 
                     judge_score: float, drpo_advantage: float) -> float:
        """
        计算防御者损失
        
        损失 = -奖励 + RAG代价 - DRPO优势
        
        Args:
            reward: 基础奖励
            rag_cost: RAG代价
            judge_score: 法官评分
            drpo_advantage: DRPO优势
            
        Returns:
            损失值
        """
        loss = (
            -self.drpo.drpo_weight * reward +
            self.drpo.rag_weight * rag_cost -
            self.drpo.drpo_weight * drpo_advantage -
            self.drpo.judge_weight * judge_score
        )
        
        return loss
    
    def get_statistics(self) -> Dict:
        """获取防御者统计信息"""
        return {
            "model_id": self.model_id,
            "total_defenses": self.total_defenses,
            "average_score": self.average_score,
            "rag_usage_count": self.rag_usage_count,
            "trainable_params": self.get_trainable_params(),
            "total_params": self.get_total_params(),
        }
