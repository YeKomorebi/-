"""
QLoRA训练器 - 管理QLoRA微调过程
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from peft import get_peft_model
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SafetyDataset(Dataset):
    """安全训练数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入
        text = f"{item.get('prompt', '')}\n{item.get('response', '')}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten().clone(),
        }


class QLoRATrainer:
    """QLoRA训练器"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def train(self, train_data: List[Dict], epochs: int = 1,
              output_dir: str = "./output") -> Dict:
        """
        训练模型
        
        Args:
            train_data: 训练数据
            epochs: 训练轮数
            output_dir: 输出目录
            
        Returns:
            训练结果
        """
        # 创建数据集
        dataset = SafetyDataset(train_data, self.tokenizer)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.model.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=self.config.model.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.config.model.LEARNING_RATE,
            warmup_ratio=self.config.model.WARMUP_RATIO,
            optim=self.config.model.OPTIM,
            logging_steps=10,
            save_steps=100,
            fp16=True,
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # 开始训练
        logger.info("开始QLoRA训练...")
        train_result = trainer.train()
        
        # 保存模型
        self.model.save_pretrained(output_dir)
        
        return {
            "training_loss": train_result.training_loss,
            "global_step": train_result.global_step,
        }
    
    def extra_training(self, model, train_data: List[Dict], 
                       epochs: int = 2) -> Dict:
        """
        额外训练（用于未及格模型）
        
        Args:
            model: 需要额外训练的模型
            train_data: 训练数据
            epochs: 额外训练轮数
            
        Returns:
            训练结果
        """
        dataset = SafetyDataset(train_data, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir="./extra_training_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.model.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=self.config.model.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.config.model.LEARNING_RATE * 0.5,  # 降低学习率
            warmup_ratio=0.05,
            optim=self.config.model.OPTIM,
            logging_steps=5,
            fp16=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        logger.info("开始额外训练...")
        train_result = trainer.train()
        
        return {
            "training_loss": train_result.training_loss,
            "global_step": train_result.global_step,
        }
