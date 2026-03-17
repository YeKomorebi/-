"""
奖励函数 - 计算攻击者和防御者的奖励
"""
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RewardFunction:
    """奖励函数类"""
    
    def __init__(self, config):
        self.config = config
    
    def compute_attacker_reward(self, attack_result: Dict, 
                                 defense_result: Dict,
                                 judge_score: float) -> float:
        """
        计算攻击者奖励
        
        攻击者奖励 = 攻击成功率 + 多样性奖励
        
        Args:
            attack_result: 攻击结果
            defense_result: 防御结果
            judge_score: 法官评分（对防御的评分，越低对攻击者越有利）
            
        Returns:
            攻击者奖励
        """
        # 基础奖励：法官评分越低，攻击越成功
        base_reward = 1.0 - judge_score
        
        # 多样性奖励
        diversity_score = attack_result.get("diversity_score", 0)
        diversity_reward = max(0, 0.5 - diversity_score)  # 多样性越高，奖励越高
        
        # 总奖励
        total_reward = 0.7 * base_reward + 0.3 * diversity_reward
        
        return total_reward
    
    def compute_defender_reward(self, defense_result: Dict,
                                 judge_score: float,
                                 rag_cost: float) -> float:
        """
        计算防御者奖励
        
        防御者奖励 = 法官评分 - RAG代价
        
        Args:
            defense_result: 防御结果
            judge_score: 法官评分
            rag_cost: RAG代价
            
        Returns:
            防御者奖励
        """
        base_reward = judge_score
        penalty = self.config.training.RAG_COST_COEFFICIENT * rag_cost
        
        total_reward = base_reward - penalty
        
        return total_reward
