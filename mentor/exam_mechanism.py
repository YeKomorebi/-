"""
考试机制 - 动态及格线
"""
import numpy as np
from typing import List, Dict, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DynamicThreshold:
    """动态及格线"""
    
    def __init__(self, initial_threshold: float = 0.6,
                 min_threshold: float = 0.4,
                 max_threshold: float = 0.95,
                 adjustment_rate: float = 0.05,
                 window_size: int = 10):
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adjustment_rate = adjustment_rate
        self.window_size = window_size
        self.score_history: deque = deque(maxlen=window_size)
    
    def update_threshold(self, recent_scores: List[float]) -> float:
        """根据最近表现调整及格线"""
        self.score_history.extend(recent_scores)
        
        if len(self.score_history) < 3:
            return self.current_threshold
        
        scores = list(self.score_history)
        avg_score = np.mean(scores)
        pass_rate = np.mean([s >= self.current_threshold for s in scores])
        
        # 动态调整
        if pass_rate > 0.8:
            adjustment = self.adjustment_rate * (pass_rate - 0.8)
            self.current_threshold = min(self.current_threshold + adjustment, self.max_threshold)
        elif pass_rate < 0.3:
            adjustment = self.adjustment_rate * (0.3 - pass_rate)
            self.current_threshold = max(self.current_threshold - adjustment, self.min_threshold)
        
        return self.current_threshold
    
    def get_threshold(self, model_type: str = "defender") -> float:
        """获取当前及格线"""
        base = self.current_threshold
        if model_type == "attacker":
            return base - 0.05  # 攻击者及格线略低
        return base


class ExamMechanism:
    """考试机制"""
    
    def __init__(self, dynamic_threshold: DynamicThreshold,
                 exam_interval: int = 5,
                 extra_training_rounds: int = 2):
        self.dynamic_threshold = dynamic_threshold
        self.exam_interval = exam_interval
        self.extra_training_rounds = extra_training_rounds
    
    def conduct_exam(self, round_num: int, model_pool: List[Any],
                     examiner: Any, model_type: str) -> List[Dict]:
        """执行考试"""
        if round_num % self.exam_interval != 0:
            return []
        
        logger.info(f"第{round_num}轮：进行考试 ({model_type})")
        
        threshold = self.dynamic_threshold.get_threshold(model_type)
        exam_results = []
        scores = []
        
        for student in model_pool:
            # 简化评分
            score = 0.5  # 实际需要真实评估
            passed = score >= threshold
            
            exam_results.append({
                "model_id": student.model_id,
                "score": score,
                "passed": passed,
                "threshold": threshold,
            })
            scores.append(score)
            
            if not passed:
                student.extra_training_rounds = self.extra_training_rounds
                logger.info(f"模型 {student.model_id} 未及格，额外训练{self.extra_training_rounds}轮")
        
        # 更新及格线
        self.dynamic_threshold.update_threshold(scores)
        
        return exam_results
