"""
基础模型类 - 所有模型的基类
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)


class BaseModel:
    """基础模型类"""
    
    def __init__(self, model_path: str, config: Any, model_id: str = "base"):
        self.model_path = model_path
        self.config = config
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        
        self._load_model()
        self._setup_tokenizer()
        
    def _load_model(self):
        """加载4-bit量化模型"""
        logger.info(f"加载模型: {self.model_path}")
        
        try:
            # 4-bit量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.LOAD_IN_4BIT,
                bnb_4bit_compute_dtype=getattr(torch, self.config.BNB_4BIT_COMPUTE_DTYPE),
                bnb_4bit_quant_type=self.config.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_use_double_quant=self.config.BNB_4BIT_USE_DOUBLE_QUANT,
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=getattr(torch, self.config.BNB_4BIT_COMPUTE_DTYPE),
            )
            
            # 准备k-bit训练
            self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info(f"模型加载完成: {self.model_path}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _setup_tokenizer(self):
        """设置分词器"""
        logger.info("设置分词器...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="right",
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.tokenizer.padding_side = "right"
            logger.info("分词器设置完成")
        except Exception as e:
            logger.error(f"分词器设置失败: {e}")
            raise
    
    def setup_lora(self):
        """配置LoRA适配器"""
        logger.info(f"配置LoRA适配器 (r={self.config.LORA_R})")
        
        try:
            self.lora_config = LoraConfig(
                r=self.config.LORA_R,
                lora_alpha=self.config.LORA_ALPHA,
                lora_dropout=self.config.LORA_DROPOUT,
                target_modules=self.config.TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("LoRA配置完成")
        except Exception as e:
            logger.error(f"LoRA配置失败: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """生成文本"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.MAX_SEQ_LENGTH - max_length,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return ""
    
    def save_model(self, save_path: str):
        """保存模型"""
        logger.info(f"保存模型到: {save_path}")
        try:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            raise
    
    def load_model(self, load_path: str):
        """加载已保存的模型"""
        logger.info(f"从 {load_path} 加载模型")
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, load_path)
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.model.parameters())
