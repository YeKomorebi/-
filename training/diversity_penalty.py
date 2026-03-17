"""
多样性惩罚机制 - 防止策略过早收敛
"""
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class DiversityPenalty:
    """多样性惩罚类"""
    
    def __init__(self, penalty_weight: float = 0.15, 
                 decay_rate: float = 0.98,
                 min_penalty: float = 0.05, 
                 history_window: int = 50,
                 similarity_threshold: float = 0.7):
        """
        初始化多样性惩罚
        
        Args:
            penalty_weight: 惩罚权重
            decay_rate: 衰减速率
            min_penalty: 最小惩罚值
            history_window: 历史窗口大小
            similarity_threshold: 相似度阈值
        """
        self.penalty_weight = penalty_weight
        self.decay_rate = decay_rate
        self.min_penalty = min_penalty
        self.history_window = history_window
        self.similarity_threshold = similarity_threshold
        
        # 历史嵌入
        self.history_embeddings: List[np.ndarray] = []
        
        # 加载句子嵌入模型（用于语义相似度计算）
        self.embed_model: Optional[SentenceTransformer] = None
        try:
            self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("句子嵌入模型加载成功")
        except Exception as e:
            logger.warning(f"句子嵌入模型加载失败: {e}，将使用简单的文本相似度")
    
    def compute_penalty(self, text: str) -> float:
        """
        计算多样性惩罚
        
        Args:
            text: 当前文本
            
        Returns:
            惩罚值 (0-1)
        """
        if not self.history_embeddings:
            return self.min_penalty
        
        # 获取当前文本嵌入
        current_embedding = self._get_embedding(text)
        if current_embedding is None:
            return self.min_penalty
        
        # 计算与历史回答的平均相似度
        similarities = []
        for hist_emb in self.history_embeddings[-self.history_window:]:
            sim = self._cosine_similarity(current_embedding, hist_emb)
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # 相似度越高，惩罚越大
        if avg_similarity > self.similarity_threshold:
            similarity_penalty = (avg_similarity - self.similarity_threshold) * self.penalty_weight
        else:
            similarity_penalty = 0.0
        
        # 词汇多样性惩罚
        vocab_penalty = self._compute_vocab_diversity(text)
        
        # 总惩罚
        total_penalty = 0.6 * similarity_penalty + 0.4 * vocab_penalty
        
        # 应用衰减
        effective_penalty = max(total_penalty, self.min_penalty)
        
        # 更新历史
        self._update_history(current_embedding)
        
        return effective_penalty
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本嵌入"""
        if self.embed_model is not None:
            try:
                embedding = self.embed_model.encode(text, convert_to_numpy=True)
                return embedding / np.linalg.norm(embedding)
            except Exception as e:
                logger.warning(f"嵌入计算失败: {e}")
                return self._simple_embedding(text)
        else:
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """简单的词袋嵌入（后备方案）"""
        words = text.lower().split()
        embedding = np.zeros(100)
        for i, word in enumerate(words[:100]):
            embedding[i] = hash(word) % 1000 / 1000.0
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    
    def _compute_vocab_diversity(self, text: str) -> float:
        """计算词汇多样性惩罚"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_ratio = len(set(words)) / len(words)
        return max(0, 0.8 - unique_ratio) * 0.1
    
    def _update_history(self, embedding: np.ndarray):
        """更新历史嵌入"""
        self.history_embeddings.append(embedding)
        if len(self.history_embeddings) > self.history_window:
            self.history_embeddings.pop(0)
    
    def reset(self):
        """重置历史"""
        self.history_embeddings = []
