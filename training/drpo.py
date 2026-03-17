"""
DRPO (Defense Robust Preference Optimization) - 防御鲁棒偏好优化
"""
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DRPO:
    """DRPO类"""
    
    def __init__(self, drpo_weight: float = 0.4, 
                 judge_weight: float = 0.4,
                 rag_weight: float = 0.2):
        """
        初始化DRPO
        
        Args:
            drpo_weight: DRPO权重
            judge_weight: 法官评分权重
            rag_weight: RAG代价权重
        """
        self.drpo_weight = drpo_weight
        self.judge_weight = judge_weight
        self.rag_weight = rag_weight
    
    def compute_advantage(self, responses: List[str], 
                          judge_scores: List[float],
                          reference_response: str = None) -> float:
        """
        计算DRPO优势
        
        Args:
            responses: 多个候选回答
            judge_scores: 法官对每个回答的评分
            reference_response: 参考回答（可选）
            
        Returns:
            优势值
        """
        if not responses or not judge_scores:
            return 0.0
        
        # 部分优势计算
        max_score = max(judge_scores)
        min_score = min(judge_scores)
        
        if max_score == min_score:
            pairwise_advantage = 0.0
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) 
                                 for s in judge_scores]
            pairwise_advantage = np.mean(normalized_scores)
        
        # 参考回答奖励
        if reference_response:
            reference_bonus = self._compute_reference_bonus(
                responses, judge_scores, reference_response
            )
        else:
            reference_bonus = 0.0
        
        # 最终优势
        advantage = pairwise_advantage + 0.2 * reference_bonus
        
        # 限制优势范围
        advantage = np.clip(advantage, -1.0, 1.0)
        
        return advantage
    
    def _compute_reference_bonus(self, responses: List[str], 
                                  judge_scores: List[float],
                                  reference: str) -> float:
        """计算参考回答奖励"""
        best_idx = np.argmax(judge_scores)
        best_response = responses[best_idx]
        similarity = self._text_similarity(best_response, reference)
        return similarity * judge_scores[best_idx]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def compute_reward(self, base_reward: float, drpo_advantage: float,
                       judge_score: float, rag_cost: float) -> float:
        """
        计算最终奖励
        
        Args:
            base_reward: 基础奖励
            drpo_advantage: DRPO优势
            judge_score: 法官评分
            rag_cost: RAG代价
            
        Returns:
            最终奖励
        """
        reward = (
            self.drpo_weight * base_reward +
            self.drpo_weight * drpo_advantage +
            self.judge_weight * judge_score -
            self.rag_weight * rag_cost
        )
        
        return reward
