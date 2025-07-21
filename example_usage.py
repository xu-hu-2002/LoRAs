# 完整使用示例 - 基于HuggingFace PEFT的各种参数高效微调技术
# 演示LoRA、DoRA、QLoRA、PiSSA的使用方法

import torch
import os
import sys

# 添加项目路径
sys.path.append('.')

# 导入配置
from config import get_model, get_peft_config, MODELS

# 导入各个技术的模块
from LoRA.lora_hf import create_lora_config, load_model_with_lora
from DoRA.dora_hf import create_dora_config, load_model_with_dora  
from QLoRA.qlora_hf import create_qlora_config, load_model_with_qlora, create_bnb_config
from PiSSA.pissa_hf import create_pissa_config, load_model_with_pissa

# 导入AdaLoRA（独立实现）
from AdaLoRA.adalora_config import AdaLoRAConfig

# 导入共享工具
from common.data_preprocessing import create_qa_dataset, create_sample_qa_dataset
from common.training_utils import create_training_arguments, train_model, save_model_and_adapter

def demo_lora():
    """演示LoRA使用"""
    print("="*60)
    print("LoRA 演示")
    print("="*60)
    
    # 1. 创建LoRA配置
    lora_config = create_lora_config(
        r=64,
        lora_alpha=128,  # 2 * r
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print("LoRA配置创建完成")
    print(f"  Rank: {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    # 注意：实际使用时替换为真实模型
    # model, tokenizer = load_model_with_lora("meta-llama/Llama-3.2-3B", lora_config)
    print("\n实际使用:")
    print("model, tokenizer = load_model_with_lora('meta-llama/Llama-3.2-3B', lora_config)")

def demo_dora():
    """演示DoRA使用"""
    print("="*60)
    print("DoRA 演示")
    print("="*60)
    
    # 1. 创建DoRA配置
    dora_config = create_dora_config(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        use_dora=True,  # 关键：启用DoRA
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print("DoRA配置创建完成")
    print(f"  Rank: {dora_config.r}")
    print(f"  Alpha: {dora_config.lora_alpha}")
    print(f"  Use DoRA: {dora_config.use_dora}")
    print(f"  Target modules: {dora_config.target_modules}")
    
    print("\n实际使用:")
    print("model, tokenizer = load_model_with_dora('meta-llama/Llama-3.2-3B', dora_config)")

def demo_qlora():
    """演示QLoRA使用"""
    print("="*60)
    print("QLoRA 演示")
    print("="*60)
    
    # 1. 创建量化配置
    bnb_config = create_bnb_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # 2. 创建QLoRA配置
    qlora_config = create_qlora_config(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print("QLoRA配置创建完成")
    print(f"  4bit量化: {bnb_config.load_in_4bit}")
    print(f"  量化类型: {bnb_config.bnb_4bit_quant_type}")
    print(f"  LoRA Rank: {qlora_config.r}")
    print(f"  LoRA Alpha: {qlora_config.lora_alpha}")
    
    print("\n实际使用:")
    print("model, tokenizer = load_model_with_qlora('meta-llama/Llama-3.2-3B', qlora_config, bnb_config)")

def demo_pissa():
    """演示PiSSA使用"""
    print("="*60)
    print("PiSSA 演示")
    print("="*60)
    
    # 1. 创建PiSSA配置
    pissa_config = create_pissa_config(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,  # PiSSA推荐设为0
        init_lora_weights="pissa_niter_4",  # 快速SVD初始化
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print("PiSSA配置创建完成")
    print(f"  Rank: {pissa_config.r}")
    print(f"  Alpha: {pissa_config.lora_alpha}")
    print(f"  Dropout: {pissa_config.lora_dropout}")
    print(f"  初始化方法: {pissa_config.init_lora_weights}")
    
    print("\n实际使用:")
    print("model, tokenizer = load_model_with_pissa('meta-llama/Llama-3.2-3B', pissa_config)")

def demo_adalora():
    """演示AdaLoRA使用"""
    print("="*60)
    print("AdaLoRA 演示")
    print("="*60)
    
    # AdaLoRA需要独立实现，不完全集成在PEFT中
    adalora_config = AdaLoRAConfig(
        r=12,
        target_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        init_warmup=500,
        final_warmup=1500,
        mask_interval=10,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.1
    )
    
    print("AdaLoRA配置创建完成")
    print(f"  初始Rank: {adalora_config.r}")
    print(f"  目标Rank: {adalora_config.target_rank}")
    print(f"  Alpha: {adalora_config.lora_alpha}")
    print(f"  预热步数: {adalora_config.init_warmup} - {adalora_config.final_warmup}")
    
    print("\n注意: AdaLoRA需要特殊的训练循环和RankAllocator")
    print("请参考AdaLoRA/adalora.py中的完整实现")

def demo_data_preprocessing():
    """演示数据预处理"""
    print("="*60)
    print("数据预处理演示")
    print("="*60)
    
    from transformers import AutoTokenizer
    
    # 模拟tokenizer（实际使用时应该是真实的tokenizer）
    print("创建示例数据集...")
    
    # 创建示例数据集
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    # dataset = create_sample_qa_dataset(tokenizer, num_samples=100)
    
    print("数据集创建方法:")
    print("1. 示例数据: create_sample_qa_dataset(tokenizer, num_samples=100)")
    print("2. TriviaQA: create_qa_dataset('triviaqa', tokenizer, max_samples=1000)")
    print("3. NQ Open: create_qa_dataset('natural_questions', tokenizer, max_samples=1000)")

def demo_training():
    """演示训练流程"""
    print("="*60)
    print("训练流程演示")
    print("="*60)
    
    # 创建训练参数
    training_args = create_training_arguments(
        output_dir="./demo_results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100
    )
    
    print("训练参数创建完成")
    print(f"  输出目录: {training_args.output_dir}")
    print(f"  训练轮数: {training_args.num_train_epochs}")
    print(f"  学习率: {training_args.learning_rate}")
    
    print("\n完整训练流程:")
    print("1. 选择PEFT技术并创建配置")
    print("2. 加载模型: model, tokenizer = load_model_with_xxx(model_name, config)")
    print("3. 准备数据: dataset = create_qa_dataset('triviaqa', tokenizer)")
    print("4. 训练模型: trainer = train_model(model, tokenizer, dataset, training_args=training_args)")
    print("5. 保存模型: save_model_and_adapter(model, tokenizer, './saved_model')")

def compare_techniques():
    """对比各种技术"""
    print("="*60)
    print("PEFT技术对比")
    print("="*60)
    
    comparison = """
    技术对比:
    
    1. LoRA (Low-Rank Adaptation)
       - 优势: 简单稳定，广泛支持
       - 劣势: 固定秩，参数分配可能不最优
       - 适用: 快速原型和一般用途
       - 显存: 基础模型大小
    
    2. DoRA (Weight-Decomposed Low-Rank Adaptation)
       - 优势: 性能优于LoRA，特别是低秩时
       - 劣势: 略微复杂，计算开销稍大
       - 适用: 性能要求较高的场景
       - 显存: 基础模型大小
    
    3. QLoRA (4-bit Quantized LoRA)
       - 优势: 显存节省65%，性能损失极小
       - 劣势: 需要支持4bit量化的硬件
       - 适用: 显存受限的环境
       - 显存: 基础模型的35%
    
    4. PiSSA (Principal Singular Values and Singular Vectors Adaptation)
       - 优势: 收敛更快（2-5倍），性能更好
       - 劣势: 需要SVD预处理，初始化较慢
       - 适用: 追求最佳性能和效率
       - 显存: 基础模型大小
    
    5. AdaLoRA (Adaptive Budget Allocation)
       - 优势: 自适应秩分配，参数效率最优
       - 劣势: 训练复杂，需要超参数调优
       - 适用: 参数预算有限且要求最优分配
       - 显存: 基础模型大小
    
    选择建议:
    - 快速实验: LoRA
    - 性能优先: DoRA或PiSSA
    - 显存受限: QLoRA
    - 参数效率: AdaLoRA
    - 最佳实践: PiSSA + QLoRA组合
    """
    
    print(comparison)

def main():
    """主函数"""
    print("🚀 参数高效微调技术演示")
    print("基于HuggingFace PEFT库的统一实现")
    print("="*60)
    
    # 检查PEFT版本
    try:
        import peft
        print(f"PEFT版本: {peft.__version__}")
    except ImportError:
        print("⚠️ PEFT库未安装，请运行: pip install peft")
        return
    
    # 演示各种技术
    demo_lora()
    print()
    
    demo_dora()
    print()
    
    demo_qlora()
    print()
    
    demo_pissa()
    print()
    
    demo_adalora()
    print()
    
    demo_data_preprocessing()
    print()
    
    demo_training()
    print()
    
    compare_techniques()
    
    print("="*60)
    print("演示完成！")
    print("\n实际使用步骤:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 选择技术: LoRA/DoRA/QLoRA/PiSSA/AdaLoRA")
    print("3. 配置参数: create_xxx_config(...)")
    print("4. 加载模型: load_model_with_xxx(...)")
    print("5. 准备数据: create_qa_dataset(...)")
    print("6. 开始训练: train_model(...)")
    print("7. 保存模型: save_model_and_adapter(...)")

if __name__ == "__main__":
    main() 