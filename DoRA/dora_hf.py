# DoRA (Weight-Decomposed Low-Rank Adaptation) - HuggingFace PEFT实现
# DoRA是LoRA的改进版本，将权重分解为magnitude和direction

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def create_dora_config(
    r: int = 128,
    lora_alpha: int = 256,  # 默认 2 * r
    lora_dropout: float = 0.1,
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    use_dora: bool = True
):
    """
    创建DoRA配置
    
    DoRA核心思想：
    - 将权重更新分解为magnitude和direction两部分
    - Direction通过LoRA处理，magnitude通过单独的可学习参数处理
    - 通常比标准LoRA性能更好
    
    Args:
        r: LoRA rank (秩)，默认128
        lora_alpha: LoRA scaling参数，默认256 (2*r)
        lora_dropout: dropout概率
        target_modules: 目标模块列表
        bias: bias处理方式
        task_type: 任务类型
        use_dora: 是否启用DoRA，默认True
    
    Returns:
        LoraConfig: PEFT DoRA配置
    """
    if target_modules is None:
        # 现代LLM的标准target_modules（包含attention和FFN层）
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention层
            "gate_proj", "up_proj", "down_proj"       # FFN层
        ]
    
    # 自动设置alpha = 2 * r
    if lora_alpha is None or lora_alpha == 256:
        lora_alpha = 2 * r
        print(f"DoRA: 自动设置 lora_alpha = 2 * r = {lora_alpha}")
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
        use_dora=use_dora  # 关键：启用DoRA
    )

def load_model_with_dora(
    model_name: str,
    dora_config: LoraConfig = None,
    torch_dtype=torch.bfloat16,
    device_map="auto"
):
    """
    加载模型并应用DoRA
    
    Args:
        model_name: 模型名称
        dora_config: DoRA配置
        torch_dtype: 数据类型
        device_map: 设备映射
    
    Returns:
        model: 应用DoRA的模型
        tokenizer: tokenizer
    """
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 应用DoRA
    if dora_config is None:
        dora_config = create_dora_config()
    
    model = get_peft_model(model, dora_config)
    model.print_trainable_parameters()
    
    print(f"DoRA已启用: use_dora = {dora_config.use_dora}")
    
    return model, tokenizer

def save_dora_adapter(model, save_path: str):
    """保存DoRA适配器"""
    model.save_pretrained(save_path)
    print(f"DoRA适配器已保存到: {save_path}")

def load_dora_adapter(base_model_name: str, adapter_path: str):
    """加载DoRA适配器"""
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载DoRA适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model

# 使用示例
def dora_example():
    """DoRA完整使用示例"""
    print("="*60)
    print("DoRA (HuggingFace PEFT) 使用示例")
    print("="*60)
    
    # 1. 创建DoRA配置
    dora_config = create_dora_config(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        use_dora=True
    )
    
    print("DoRA配置:")
    print(f"  rank (r): {dora_config.r}")
    print(f"  alpha: {dora_config.lora_alpha}")
    print(f"  dropout: {dora_config.lora_dropout}")
    print(f"  use_dora: {dora_config.use_dora}")
    print(f"  scaling factor: {dora_config.lora_alpha / dora_config.r}")
    print(f"  target_modules: {dora_config.target_modules}")
    
    print("\nDoRA vs LoRA:")
    print("  LoRA: W = W₀ + α/r * B * A")
    print("  DoRA: W = m * (W₀ + ΔW) / ||W₀ + ΔW||")
    print("  - m: magnitude参数 (可学习)")
    print("  - ΔW: direction更新 (通过LoRA)")
    
    print("\n优势:")
    print("  - 更好的性能，特别是在低rank时")
    print("  - 更稳定的训练")
    print("  - 支持Linear和Conv2D层")
    
    print("\n使用方法:")
    print("1. model, tokenizer = load_model_with_dora('meta-llama/Llama-3.2-3B', dora_config)")
    print("2. 训练模型...")
    print("3. save_dora_adapter(model, 'path/to/adapter')")
    print("4. model = load_dora_adapter('meta-llama/Llama-3.2-3B', 'path/to/adapter')")

if __name__ == "__main__":
    dora_example() 