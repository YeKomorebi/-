"""
末位淘汰机制
"""
import numpy as np
from typing import List, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EliminationMechanism:
    """末位淘汰机制"""
    
    def __init__(self, elimination_interval: int = 20, 
                 eliminate_count: int = 2,
                 score_window_size: int = 20):
        """
        初始化淘汰机制
        
        Args:
            elimination_interval: 淘汰间隔（轮数）
            eliminate_count: 每次淘汰数量
            score_window_size: 得分窗口大小
        """
        self.elimination_interval = elimination_interval
        self.eliminate_count = eliminate_count
        self.score_window_size = score_window_size
        
        # 模型得分窗口 {model_id: [scores]}
        self.score_window: Dict[str, List[float]] = defaultdict(list)
    
    def update_score(self, round_num: int, model_id: str, score: float):
        """
        更新模型得分
        
        Args:
            round_num: 当前轮数
            model_id: 模型ID
            score: 得分
        """
        self.score_window[model_id].append(score)
        
        # 保持窗口大小
        if len(self.score_window[model_id]) > self.score_window_size:
            self.score_window[model_id].pop(0)
    
    def get_elimination_candidates(self, round_num: int, 
                                    model_pool: List) -> List:
        """
        获取淘汰候选
        
        Args:
            round_num: 当前轮数
            model_pool: 模型池
            
        Returns:
            淘汰候选模型列表
        """
        if round_num % self.elimination_interval != 0:
            return []
        
        # 计算平均得分
        avg_scores = []
        for model in model_pool:
            model_id = model.model_id
            if model_id in self.score_window and len(self.score_window[model_id]) >= 10:
                avg_score = np.mean(self.score_window[model_id])
                avg_scores.append((model, avg_score))
        
        if len(avg_scores) <= self.eliminate_count:
            logger.warning("模型池太小，无法进行淘汰")
            return []
        
        # 按得分排序（低分在前）
        avg_scores.sort(key=lambda x: x[1])
        
        # 返回最差的模型
        candidates = [model for model, _ in avg_scores[:self.eliminate_count]]
        
        logger.info(f"淘汰候选: {[m.model_id for m in candidates]}")
        
        return candidates
    
    def remove_models(self, model_pool: List, candidates: List) -> List:
        """
        从模型池中移除模型
        
        Args:
            model_pool: 原模型池
            candidates: 要移除的模型
            
        Returns:
            新模型池
        """
        candidate_ids = {m.model_id for m in candidates}
        new_pool = [m for m in model_pool if m.model_id not in candidate_ids]
        
        # 清理得分记录
        for model_id in candidate_ids:
            if model_id in self.score_window:
                del self.score_window[model_id]
        
        return new_pool
