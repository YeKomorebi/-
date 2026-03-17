"""
导师管理系统 - 包含任期限制和更换考试
"""
from typing import List, Optional, Any, Dict  # ✅ 添加 Dict
import logging

logger = logging.getLogger(__name__)


class MentorSystem:
    """导师管理系统"""
    
    def __init__(self, max_consecutive_terms: int = 3,
                 exam_interval: int = 10,
                 evaluation_rounds: int = 10):
        self.max_consecutive_terms = max_consecutive_terms
        self.exam_interval = exam_interval
        self.evaluation_rounds = evaluation_rounds
        
        self.current_mentor: Optional[Any] = None
        self.consecutive_terms: int = 0
        self.mentor_history: List[Any] = []
    
    def conduct_mentor_exam(self, round_num: int, 
                            candidate_pool: List[Any],
                            attacker_pool: List[Any],
                            model_type: str = "defender") -> Optional[Any]:
        """进行导师更换考试"""
        if round_num % self.exam_interval != 0:
            return self.current_mentor
        
        logger.info(f"第{round_num}轮：进行导师考试 ({model_type})")
        
        # 选出表现最好的作为备选导师
        scores = []
        for candidate in candidate_pool:
            score = self._evaluate_candidate(candidate, attacker_pool)
            scores.append((candidate, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        backup_mentor = scores[0][0] if scores else None
        
        if backup_mentor is None:
            return self.current_mentor
        
        # 备选导师 vs 现任导师 对抗测试
        if self.current_mentor is not None:
            current_score = self._mentor_evaluation_rounds(
                self.current_mentor, attacker_pool
            )
            backup_score = self._mentor_evaluation_rounds(
                backup_mentor, attacker_pool
            )
            
            logger.info(f"现任导师得分: {current_score:.4f}, 备选得分: {backup_score:.4f}")
            
            # 检查任期限制
            if self.consecutive_terms >= self.max_consecutive_terms:
                logger.info(f"导师已达最大任期 ({self.max_consecutive_terms})，强制更换")
                new_mentor = backup_mentor
                self.consecutive_terms = 1
            elif backup_score > current_score:
                logger.info("备选导师表现更好，更换导师")
                new_mentor = backup_mentor
                self.consecutive_terms += 1
            else:
                logger.info("现任导师继续任职")
                new_mentor = self.current_mentor
                self.consecutive_terms += 1
        else:
            logger.info("首次任命导师")
            new_mentor = backup_mentor
            self.consecutive_terms = 1
        
        if self.current_mentor:
            self.mentor_history.append(self.current_mentor)
        self.current_mentor = new_mentor
        
        return new_mentor
    
    def _evaluate_candidate(self, candidate: Any, 
                            attacker_pool: List[Any]) -> float:
        """评估候选导师"""
        total_score = 0
        num_evaluations = min(5, len(attacker_pool))
        
        for attacker in attacker_pool[:num_evaluations]:
            score = 0.5  # 简化评估
            total_score += score
        
        return total_score / num_evaluations if num_evaluations > 0 else 0.0
    
    def _mentor_evaluation_rounds(self, mentor: Any, 
                                   attacker_pool: List[Any],
                                   rounds: int = None) -> float:
        """导师对抗评估"""
        if rounds is None:
            rounds = self.evaluation_rounds
        
        total_score = 0
        for i in range(rounds):
            if attacker_pool:
                attacker = attacker_pool[i % len(attacker_pool)]
                score = 0.5  # 简化评估
                total_score += score
        
        return total_score / rounds if rounds > 0 else 0.0
    
    def get_mentor_info(self) -> Dict:
        """获取导师信息"""
        return {
            "current_mentor_id": self.current_mentor.model_id if self.current_mentor else None,
            "consecutive_terms": self.consecutive_terms,
            "max_terms": self.max_consecutive_terms,
            "mentor_history_count": len(self.mentor_history),
        }
