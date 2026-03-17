"""
模型配置 - 定义攻击者、防御者、法官的模型参数
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ModelConfig:
    """模型配置类"""
    
    # ========== 基础模型路径 ==========
    # 注意：Qwen2.5没有精确的2.5B，使用1.5B或3B替代
    ATTACKER_MODEL_PATH: str = "Qwen/Qwen2.5-1.5B-Instruct"
    DEFENDER_MODEL_PATH: str = "Qwen/Qwen2.5-1.5B-Instruct"
    JUDGE_MODEL_PATH: str = "Qwen/Qwen2.5-3B-Instruct"  # 法官用更大的模型
    
    # ========== QLoRA配置 ==========
    LOAD_IN_4BIT: bool = True
    BNB_4BIT_COMPUTE_DTYPE: str = "float16"
    BNB_4BIT_QUANT_TYPE: str = "nf4"
    BNB_4BIT_USE_DOUBLE_QUANT: bool = True
    
    # LoRA参数
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    TARGET_MODULES: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # ========== 训练配置 ==========
    LEARNING_RATE: float = 2e-4
    PER_DEVICE_TRAIN_BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 8
    MAX_STEPS: int = 1000
    WARMUP_RATIO: float = 0.1
    
    # ========== 优化器配置 ==========
    OPTIM: str = "paged_adamw_8bit"
    
    # ========== 设备配置 ==========
    DEVICE: str = "cuda"
    MAX_SEQ_LENGTH: int = 2048
    
    def __post_init__(self):
        """初始化后验证"""
        assert self.LORA_ALPHA >= self.LORA_R, "lora_alpha应>=lora_r"
        assert 0 < self.LORA_DROPOUT < 1, "dropout应在0-1之间"
        assert self.LEARNING_RATE > 0, "学习率必须为正"
