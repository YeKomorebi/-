"""
指标追踪
"""
from typing import Dict, List
from collections import defaultdict
import json
from pathlib import Path


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self, save_path: str = "./logs/metrics.json"):
        self.save_path = Path(save_path)
        self.metrics: Dict[str, List[float]] = defaultdict(list)
    
    def log(self, name: str, value: float, round_num: int = None):
        """记录指标"""
        self.metrics[name].append(value)
    
    def get_average(self, name: str, window: int = None) -> float:
        """获取平均值"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        values = self.metrics[name]
        if window:
            values = values[-window:]
        
        return sum(values) / len(values)
    
    def save(self):
        """保存指标"""
        with open(self.save_path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
    
    def load(self):
        """加载指标"""
        if self.save_path.exists():
            with open(self.save_path, 'r') as f:
                self.metrics = defaultdict(list, json.load(f))
