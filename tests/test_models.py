"""
模型测试
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.system_config import SystemConfig


class TestModelConfig:
    """模型配置测试"""
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = SystemConfig()
        assert config.model.LORA_R == 16
        assert config.model.LOAD_IN_4BIT == True
    
    def test_config_validation(self):
        """测试配置验证"""
        config = SystemConfig()
        assert config.training.validate() == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
