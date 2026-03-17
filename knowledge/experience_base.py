"""
历史经验库 - 高质量Q&A记录
"""
from typing import List, Dict
from pathlib import Path
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


class ExperienceBase:
    """历史经验库"""
    
    def __init__(self, save_path: str = "./knowledge_base/experience_base.json",
                 max_size: int = 1000):
        self.save_path = Path(save_path)
        self.max_size = max_size
        self.experiences: deque = deque(maxlen=max_size)
        self._load()
    
    def add_experience(self, question: str, answer: str, 
                       quality_score: float, model_id: str):
        """添加经验"""
        experience = {
            "question": question,
            "answer": answer,
            "quality_score": quality_score,
            "model_id": model_id,
        }
        self.experiences.append(experience)
        self._save()
    
    def get_high_quality(self, threshold: float = 0.8, 
                         limit: int = 100) -> List[Dict]:
        """获取高质量经验"""
        high_quality = [e for e in self.experiences 
                       if e["quality_score"] >= threshold]
        return list(high_quality)[:limit]
    
    def _save(self):
        """保存"""
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.experiences), f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """加载"""
        if self.save_path.exists():
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.experiences = deque(data, maxlen=self.max_size)
