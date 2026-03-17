"""
系统配置 - 整合所有配置
"""
from dataclasses import dataclass, field
from pathlib import Path
from .model_config import ModelConfig
from .training_config import TrainingConfig
from typing import Dict  # ✅ 添加导入


@dataclass
class SystemConfig:
    """系统总配置"""
    
    # 路径配置
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    MODEL_SAVE_DIR: Path = field(default_factory=lambda: Path("./saved_models"))
    LOG_DIR: Path = field(default_factory=lambda: Path("./logs"))
    KNOWLEDGE_BASE_DIR: Path = field(default_factory=lambda: Path("./knowledge_base"))
    
    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    USE_WANDB: bool = False
    WANDB_PROJECT: str = "adversarial-safety"
    
    # 随机种子
    RANDOM_SEED: int = 42
    
    # 知识库配置
    TRUTH_BASE_INIT_SIZE: int = 100
    EXPERIENCE_BASE_MAX_SIZE: int = 1000
    CREATIVITY_BASE_MAX_SIZE: int = 500
    CREATIVITY_THRESHOLD: float = 0.8
    
    def __post_init__(self):
        """初始化后创建目录"""
        self.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_config_dict(self) -> Dict:
        """获取配置字典（用于日志记录）"""
        return {
            "model": {
                "attacker_path": self.model.ATTACKER_MODEL_PATH,
                "defender_path": self.model.DEFENDER_MODEL_PATH,
                "judge_path": self.model.JUDGE_MODEL_PATH,
                "lora_r": self.model.LORA_R,
                "learning_rate": self.model.LEARNING_RATE,
            },
            "training": {
                "num_rounds": self.training.NUM_ROUNDS,
                "diversity_weight": self.training.DIVERSITY_PENALTY_WEIGHT,
                "max_mentor_terms": self.training.MAX_MENTOR_TERMS,
                "exam_interval": self.training.EXAM_INTERVAL,
                "elimination_interval": self.training.ELIMINATION_INTERVAL,
            },
            "system": {
                "random_seed": self.RANDOM_SEED,
                "use_wandb": self.USE_WANDB,
            }
        }


# 默认配置实例
DEFAULT_CONFIG = SystemConfig()
