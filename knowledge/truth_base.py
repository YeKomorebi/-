"""
真理库 - 基本规则
"""
from typing import List, Dict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class TruthBase:
    """真理库"""
    
    def __init__(self, save_path: str = "./knowledge_base/truth_base.json"):
        self.save_path = Path(save_path)
        self.truths: List[Dict] = []
        self._initialize()
    
    def _initialize(self):
        """初始化真理库"""
        # 基本安全规则
        default_truths = [
            {"id": 1, "rule": "不得生成有害内容", "category": "safety"},
            {"id": 2, "rule": "不得泄露隐私信息", "category": "privacy"},
            {"id": 3, "rule": "不得进行歧视性言论", "category": "fairness"},
            {"id": 4, "rule": "必须提供准确信息", "category": "accuracy"},
            {"id": 5, "rule": "必须尊重用户", "category": "respect"},
        ]
        
        if self.save_path.exists():
            self._load()
        else:
            self.truths = default_truths
            self._save()
        
        logger.info(f"真理库初始化完成，共{len(self.truths)}条规则")
    
    def add_truth(self, rule: str, category: str) -> int:
        """添加真理"""
        new_id = max([t["id"] for t in self.truths], default=0) + 1
        truth = {"id": new_id, "rule": rule, "category": category}
        self.truths.append(truth)
        self._save()
        return new_id
    
    def get_truths(self, category: str = None) -> List[Dict]:
        """获取真理"""
        if category:
            return [t for t in self.truths if t["category"] == category]
        return self.truths
    
    def _save(self):
        """保存到文件"""
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.truths, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """从文件加载"""
        with open(self.save_path, 'r', encoding='utf-8') as f:
            self.truths = json.load(f)
