"""
主程序入口
"""
import argparse
import sys
from pathlib import Path
from typing import List, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config.system_config import SystemConfig
from models.attacker import AttackerModel
from models.defender import DefenderModel
from models.judge import JudgeModel
from training.reward_function import RewardFunction
from evolution.elimination import EliminationMechanism
from evolution.hybridization import HybridizationModule
from evolution.mutation import MutationGenerator
from mentor.mentor_system import MentorSystem
from mentor.exam_mechanism import ExamMechanism, DynamicThreshold
from knowledge.truth_base import TruthBase
from knowledge.experience_base import ExperienceBase
from knowledge.creativity_base import CreativityBase
from rag.rag_system import RAGSystem
from utils.logger import setup_logger
from utils.metrics import MetricsTracker
from utils.helpers import set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="对抗性多智能体AI安全训练框架")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "test"], help="运行模式")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--use_wandb", action="store_true", help="使用wandb")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test", action="store_true", help="测试模式（不加载真实模型）")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化配置
    config = SystemConfig()
    config.RANDOM_SEED = args.seed
    config.USE_WANDB = args.use_wandb
    config.training.NUM_ROUNDS = args.epochs
    
    # 设置日志
    logger = setup_logger("adversarial_safety", str(config.LOG_DIR), config.LOG_LEVEL)
    logger.info("=" * 60)
    logger.info("对抗性多智能体AI安全训练框架")
    logger.info("=" * 60)
    logger.info(f"配置: {config.get_config_dict()}")
    
    # 初始化指标追踪
    metrics = MetricsTracker(str(config.LOG_DIR / "metrics.json"))
    
    try:
        # 先初始化知识库
        logger.info("初始化知识库...")
        truth_base = TruthBase(str(config.KNOWLEDGE_BASE_DIR / "truth_base.json"))
        experience_base = ExperienceBase(
            str(config.KNOWLEDGE_BASE_DIR / "experience_base.json"),
            config.EXPERIENCE_BASE_MAX_SIZE
        )
        creativity_base = CreativityBase(
            str(config.KNOWLEDGE_BASE_DIR / "creativity_base.json"),
            config.CREATIVITY_BASE_MAX_SIZE,
            config.CREATIVITY_THRESHOLD
        )
        
        # 初始化模型
        logger.info("初始化模型...")
        
        if args.test:
            # 测试模式：使用模拟模型
            logger.info("测试模式：使用模拟模型")
            attackers = [MockAttackerModel(f"attacker_{i}") for i in range(config.training.NUM_ATTACKERS)]
            defenders = [MockDefenderModel(f"defender_{i}") for i in range(config.training.NUM_DEFENDERS)]
            judge = MockJudgeModel()
        else:
            # 真实模式
            attackers = [
                AttackerModel(config.model.ATTACKER_MODEL_PATH, config, f"attacker_{i}")
                for i in range(config.training.NUM_ATTACKERS)
            ]
            defenders = [
                DefenderModel(config.model.DEFENDER_MODEL_PATH, config, f"defender_{i}")
                for i in range(config.training.NUM_DEFENDERS)
            ]
            judge = JudgeModel(
                config.model.JUDGE_MODEL_PATH, 
                config, 
                model_id="judge",
                truth_base=truth_base,
                experience_base=experience_base,
                creativity_base=creativity_base,
            )
        
        # 初始化组件
        reward_fn = RewardFunction(config)
        elimination = EliminationMechanism(
            config.training.ELIMINATION_INTERVAL,
            config.training.ELIMINATION_COUNT,
        )
        hybridization = HybridizationModule(alpha=config.training.HYBRID_ALPHA)
        mutation = MutationGenerator(noise_scale=config.training.MUTATION_NOISE_SCALE)
        
        dynamic_threshold = DynamicThreshold(
            config.training.INITIAL_THRESHOLD,
            config.training.MIN_THRESHOLD,
            config.training.MAX_THRESHOLD,
            config.training.THRESHOLD_ADJUSTMENT_RATE,
            config.training.THRESHOLD_WINDOW_SIZE,
        )
        
        exam_mechanism = ExamMechanism(
            dynamic_threshold,
            config.training.EXAM_INTERVAL,
            config.training.EXTRA_TRAINING_ROUNDS,
        )
        
        attacker_mentor = MentorSystem(
            config.training.MAX_MENTOR_TERMS,
            config.training.MENTOR_EXAM_INTERVAL,
            config.training.MENTOR_EVALUATION_ROUNDS,
        )
        defender_mentor = MentorSystem(
            config.training.MAX_MENTOR_TERMS,
            config.training.MENTOR_EXAM_INTERVAL,
            config.training.MENTOR_EVALUATION_ROUNDS,
        )
        
        rag_system = RAGSystem(
            config.training.RAG_COST_COEFFICIENT,
            config.training.STAGNATION_WINDOW,
            config.training.RAG_COST_DECAY_RATE,
        )
        
        # 训练循环
        logger.info(f"开始训练，共{config.training.NUM_ROUNDS}轮")
        
        for round_num in range(1, config.training.NUM_ROUNDS + 1):
            logger.info(f"\n{'='*40}")
            logger.info(f"第 {round_num} 轮")
            logger.info(f"{'='*40}")
            
            round_scores = []
            
            # 攻击者-防御者对抗
            for attacker in attackers:
                attacks = attacker.generate_attack(num_questions=2)
                
                for attack in attacks:
                    defender = defenders[round_num % len(defenders)]
                    
                    use_rag = (round_num % 3 == 0)
                    defense = defender.defend(attack["question"], use_rag=use_rag)
                    
                    judge_score = judge.single_evaluate(
                        attack["question"], 
                        defense["best_response"]
                    )
                    
                    attacker_reward = reward_fn.compute_attacker_reward(
                        attack, defense, judge_score
                    )
                    defender_reward = reward_fn.compute_defender_reward(
                        defense, judge_score, defense["rag_cost"]
                    )
                    
                    attacker.update_success_rate(judge_score < 0.5)
                    defender.update_score(judge_score)
                    
                    metrics.log("attacker_reward", attacker_reward, round_num)
                    metrics.log("defender_reward", defender_reward, round_num)
                    metrics.log("judge_score", judge_score, round_num)
                    
                    round_scores.append(judge_score)
                    
                    # 更新知识库
                    if judge_score >= 0.8:
                        judge.add_to_experience_base(
                            attack["question"],
                            defense["best_response"],
                            judge_score,
                            defender.model_id
                        )
                    
                    creativity_indicators = ["创新", "独特", "新颖", "巧妙", "特别"]
                    has_creativity = any(ind in defense["best_response"] 
                                         for ind in creativity_indicators)
                    if has_creativity and judge_score >= 0.7:
                        creativity_score = 0.7 + (judge_score - 0.7) * 0.5
                        judge.add_to_creativity_base(
                            defense["best_response"],
                            creativity_score,
                            "answer"
                        )
            
            # 考试机制
            exam_mechanism.conduct_exam(round_num, defenders, judge, "defender")
            exam_mechanism.conduct_exam(round_num, attackers, judge, "attacker")
            
            # 导师考试
            attacker_mentor.conduct_mentor_exam(round_num, attackers, defenders, "attacker")
            defender_mentor.conduct_mentor_exam(round_num, defenders, attackers, "defender")
            
            # 进化机制
            if round_num % config.training.ELIMINATION_INTERVAL == 0:
                logger.info("进行进化操作...")
                
                elim_candidates = elimination.get_elimination_candidates(round_num, defenders)
                if elim_candidates:
                    defenders = elimination.remove_models(defenders, elim_candidates)
                    logger.info(f"淘汰 {len(elim_candidates)} 个防御者模型")
                
                if len(defenders) >= 2:
                    sorted_defenders = sorted(defenders, 
                                             key=lambda x: x.average_score, 
                                             reverse=True)
                    hybrid = hybridization.hybridize(sorted_defenders[0], sorted_defenders[1])
                    defenders.append(hybrid)
                    logger.info("创建杂交模型")
                
                if defenders:
                    best_defender = max(defenders, key=lambda x: x.average_score)
                    mutant = mutation.generate_mutant(best_defender)
                    defenders.append(mutant)
                    logger.info("创建突变模型")
            
            # 更新淘汰机制得分
            for defender in defenders:
                elimination.update_score(round_num, defender.model_id, defender.average_score)
            
            # RAG停滞检测
            rag_system.update_cost_on_stagnation(round_scores)
            
            # 保存指标
            metrics.save()
            
            avg_score = sum(round_scores) / len(round_scores) if round_scores else 0
            logger.info(f"本轮平均得分: {avg_score:.4f}")
            logger.info(f"RAG代价: {rag_system.get_cost():.4f}")
        
        # 保存最终模型
        if not args.test:
            logger.info("\n保存最终模型...")
            for i, defender in enumerate(defenders):
                save_path = config.MODEL_SAVE_DIR / f"defender_final_{i}"
                defender.save_model(str(save_path))
        
        metrics.save()
        
        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise
    finally:
        metrics.save()


# ========== 模拟模型类（用于测试） ==========
class MockAttackerModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.attack_history = []
        self.success_rate = 0.0
        self.total_attacks = 0
        self.successful_attacks = 0
        self.extra_training_rounds = 0
    
    def generate_attack(self, num_questions: int = 2) -> List[Dict]:
        attacks = []
        for i in range(num_questions):
            attack = {"question": f"测试攻击问题{i}", "diversity_score": 0.1}
            attacks.append(attack)
            self.attack_history.append(attack)
        return attacks
    
    def update_success_rate(self, success: bool):
        self.total_attacks += 1
        if success:
            self.successful_attacks += 1
        self.success_rate = self.successful_attacks / self.total_attacks if self.total_attacks > 0 else 0.0


class MockDefenderModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.defense_history = []
        self.average_score = 0.0
        self.total_defenses = 0
        self.total_score = 0.0
        self.rag_usage_count = 0
        self.extra_training_rounds = 0
    
    def defend(self, attack_question: str, use_rag: bool = False) -> Dict:
        defense = {
            "question": attack_question,
            "responses": [f"测试防御回答{i}" for i in range(3)],
            "best_response": f"测试防御回答0",
            "rag_cost": 1.0 if use_rag else 0.0,
        }
        self.defense_history.append(defense)
        self.total_defenses += 1
        if use_rag:
            self.rag_usage_count += 1
        return defense
    
    def update_score(self, score: float):
        self.total_score += score
        self.average_score = self.total_score / self.total_defenses if self.total_defenses > 0 else 0.0


class MockJudgeModel:
    def __init__(self):
        self.evaluation_history = []
        self.truth_base = TruthBase()
        self.experience_base = ExperienceBase()
        self.creativity_base = CreativityBase()
    
    def single_evaluate(self, question: str, response: str) -> float:
        import random
        score = random.uniform(0.5, 0.9)
        self.evaluation_history.append({"question": question, "score": score})
        return score
    
    def add_to_experience_base(self, question: str, answer: str, quality_score: float, model_id: str):
        self.experience_base.add_experience(question, answer, quality_score, model_id)
    
    def add_to_creativity_base(self, content: str, creativity_score: float, content_type: str = "answer"):
        self.creativity_base.add_creative_item(content, creativity_score, content_type)


if __name__ == "__main__":
    main()
