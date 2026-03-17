"""
RAG检索系统
"""
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG检索系统"""
    
    def __init__(self, cost_coefficient: float = 0.2,
                 stagnation_window: int = 5,
                 decay_rate: float = 0.8):
        self.cost_coefficient = cost_coefficient
        self.stagnation_window = stagnation_window
        self.decay_rate = decay_rate
        
        self.retrieval_count: int = 0
        self.score_history: List[float] = []
    
    def retrieve(self, query: str, knowledge_base: List[Dict],
                 top_k: int = 3) -> List[Dict]:
        """
        检索相关知识
        
        Args:
            query: 查询
            knowledge_base: 知识库
            top_k: 返回数量
            
        Returns:
            检索结果
        """
        if not knowledge_base:
            return []
        
        # 简化检索（实际应使用向量检索）
        results = knowledge_base[:top_k]
        self.retrieval_count += 1
        
        logger.debug(f"RAG检索: {len(results)} 条结果")
        
        return results
    
    def get_cost(self) -> float:
        """获取当前RAG代价"""
        return self.cost_coefficient
    
    def update_cost_on_stagnation(self, scores: List[float]):
        """停滞时降低RAG代价"""
        self.score_history.extend(scores)
        
        if len(self.score_history) < self.stagnation_window:
            return
        
        recent_scores = self.score_history[-self.stagnation_window:]
        
        # 检测停滞
        if np.std(recent_scores) < 0.01 or np.mean(recent_scores) <= np.mean(self.score_history[-10:-5]):
            self.cost_coefficient *= self.decay_rate
            logger.info(f"检测到停滞，RAG代价降低至: {self.cost_coefficient:.4f}")
    
    def reset(self):
        """重置"""
        self.retrieval_count = 0
        self.score_history = []
