"""
奇思妙想库 - 创意回答记录
"""
from typing import List, Dict
from pathlib import Path
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


class CreativityBase:
    """奇思妙想库"""
    
    def __init__(self, save_path: str = "./knowledge_base/creativity_base.json",
                 max_size: int = 500,
                 creativity_threshold: float = 0.8):
        self.save_path = Path(save_path)
        self.max_size = max_size
        self.creativity_threshold = creativity_threshold
        self.creative_items: deque = deque(maxlen=max_size)
        self._load()
    
    def add_creative_item(self, content: str, creativity_score: float,
                          content_type: str = "answer"):
        """添加创意内容"""
        if creativity_score >= self.creativity_threshold:
            item = {
                "content": content,
                "creativity_score": creativity_score,
                "content_type": content_type,
            }
            self.creative_items.append(item)
            self._save()
    
    def get_creative_items(self, limit: int = 50) -> List[Dict]:
        """获取创意内容"""
        return list(self.creative_items)[:limit]
    
    def _save(self):
        """保存"""
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.creative_items), f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """加载"""
        if self.save_path.exists():
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.creative_items = deque(data, maxlen=self.max_size)
