"""
杂交模块 - 模型参数混合
"""
import torch
import copy
from typing import List, Any, Dict
import logging

logger = logging.getLogger(__name__)


class HybridizationModule:
    """杂交模块类"""
    
    def __init__(self, alpha: float = 0.5):
        """
        初始化杂交模块
        
        Args:
            alpha: 混合比例 (0-1)，0.5表示平均混合
        """
        self.alpha = alpha
        assert 0 <= alpha <= 1, "alpha必须在0-1之间"
    
    def hybridize(self, model_a: Any, model_b: Any) -> Any:
        """
        杂交两个模型
        
        Args:
            model_a: 父本模型A
            model_b: 父本模型B
            
        Returns:
            杂交后的新模型
        """
        logger.info(f"执行模型杂交 (alpha={self.alpha})")
        
        # 创建模型B的深拷贝作为子代
        child_model = copy.deepcopy(model_b)
        
        try:
            # 获取两个模型的参数
            params_a = dict(model_a.model.named_parameters())
            params_b = dict(model_b.model.named_parameters())
            params_child = dict(child_model.model.named_parameters())
            
            # 混合参数
            with torch.no_grad():
                for name in params_child.keys():
                    if name in params_a and name in params_b:
                        # 线性插值混合
                        params_child[name].data = (
                            self.alpha * params_a[name].data + 
                            (1 - self.alpha) * params_b[name].data
                        )
            
            # 设置子代模型ID
            child_model.model_id = f"hybrid_{model_a.model_id}_{model_b.model_id}"
            
            # 继承统计信息（平均）
            if hasattr(model_a, 'average_score') and hasattr(model_b, 'average_score'):
                child_model.average_score = (
                    model_a.average_score + model_b.average_score
                ) / 2
            
            if hasattr(model_a, 'total_defenses') and hasattr(model_b, 'total_defenses'):
                child_model.total_defenses = (
                    model_a.total_defenses + model_b.total_defenses
                ) // 2
            
            logger.info(f"杂交完成: {child_model.model_id}")
            
        except Exception as e:
            logger.error(f"杂交失败: {e}")
            # 如果杂交失败，返回模型B的拷贝
            child_model.model_id = f"copy_{model_b.model_id}"
        
        return child_model
    
    def hybridize_multiple(self, models: List[Any], weights: List[float] = None) -> Any:
        """
        杂交多个模型
        
        Args:
            models: 父本模型列表
            weights: 每个模型的权重（可选，默认平均）
            
        Returns:
            杂交后的新模型
        """
        if not models:
            raise ValueError("模型列表不能为空")
        
        if len(models) == 1:
            return copy.deepcopy(models[0])
        
        # 默认平均权重
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            # 归一化权重
            total = sum(weights)
            weights = [w / total for w in weights]
        
        logger.info(f"执行多模型杂交 (数量={len(models)})")
        
        # 以第一个模型为基准
        child_model = copy.deepcopy(models[0])
        
        try:
            with torch.no_grad():
                # 获取所有模型的参数
                all_params = [dict(m.model.named_parameters()) for m in models]
                child_params = dict(child_model.model.named_parameters())
                
                # 加权混合
                for name in child_params.keys():
                    mixed_data = torch.zeros_like(child_params[name].data)
                    
                    for i, params in enumerate(all_params):
                        if name in params:
                            mixed_data += weights[i] * params[name].data
                    
                    child_params[name].data = mixed_data
            
            child_model.model_id = f"hybrid_multi_{'_'.join([m.model_id for m in models])}"
            
            # 平均统计信息
            if hasattr(models[0], 'average_score'):
                child_model.average_score = sum(m.average_score for m in models) / len(models)
            
            logger.info(f"多模型杂交完成: {child_model.model_id}")
            
        except Exception as e:
            logger.error(f"多模型杂交失败: {e}")
            child_model.model_id = f"copy_{models[0].model_id}"
        
        return child_model
    
    def get_config(self) -> Dict:
        """获取配置信息"""
        return {
            "alpha": self.alpha,
            "module_type": "hybridization"
        }
