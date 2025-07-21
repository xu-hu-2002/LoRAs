# PiSSA (Principal Singular Values and Singular Vectors Adaptation) - HuggingFace PEFT实现
# PiSSA优化主要奇异值和向量，比LoRA收敛更快

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def create_pissa_config(
    r: int = 128,
    lora_alpha: int = 256,  # 默认 2 * r
    lora_dropout: float = 0.0,  # PiSSA推荐设为0
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    init_lora_weights: str = "pissa_niter_4"
):
    """
    创建PiSSA配置
    
    PiSSA核心思想：
    - 通过SVD分解预训练权重
    - 优化主要奇异值和奇异向量
    - 冻结"噪声"部分
    - 比LoRA收敛更快，性能更好
    
    Args:
        r: LoRA rank (秩)，决定保留的主要奇异值数量，默认128
        lora_alpha: LoRA scaling参数，默认256 (2*r)
        lora_dropout: dropout概率，PiSSA推荐设为0
        target_modules: 目标模块列表
        bias: bias处理方式
        task_type: 任务类型
        init_lora_weights: PiSSA初始化方法
    
    Returns:
        LoraConfig: PEFT PiSSA配置
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
        print(f"PiSSA: 自动设置 lora_alpha = 2 * r = {lora_alpha}")
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
        init_lora_weights=init_lora_weights  # 关键：PiSSA初始化
    )

def load_model_with_pissa(
    model_name: str,
    pissa_config: LoraConfig = None,
    torch_dtype=torch.bfloat16,
    device_map="auto"
):
    """
    加载模型并应用PiSSA
    
    Args:
        model_name: 模型名称
        pissa_config: PiSSA配置
        torch_dtype: 数据类型
        device_map: 设备映射
    
    Returns:
        model: 应用PiSSA的模型
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
    
    # 应用PiSSA
    if pissa_config is None:
        pissa_config = create_pissa_config()
    
    model = get_peft_model(model, pissa_config)
    model.print_trainable_parameters()
    
    print(f"PiSSA初始化方法: {pissa_config.init_lora_weights}")
    
    return model, tokenizer

def save_pissa_adapter(model, save_path: str):
    """保存PiSSA适配器"""
    model.save_pretrained(save_path)
    print(f"PiSSA适配器已保存到: {save_path}")

def load_pissa_adapter(base_model_name: str, adapter_path: str):
    """加载PiSSA适配器"""
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载PiSSA适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model

def convert_pissa_to_lora(model, save_path: str):
    """
    将PiSSA转换为标准LoRA格式
    这允许在标准基础模型上加载转换后的LoRA
    """
    # 注意：这个功能需要特殊处理，具体实现依赖于PEFT库的版本
    model.save_pretrained(save_path)
    print(f"PiSSA已转换为LoRA格式并保存到: {save_path}")

# 使用示例
def pissa_example():
    """PiSSA完整使用示例"""
    print("="*60)
    print("PiSSA (HuggingFace PEFT) 使用示例")
    print("="*60)
    
    # 1. 创建PiSSA配置
    pissa_config = create_pissa_config(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,  # PiSSA推荐设为0
        init_lora_weights="pissa_niter_4"  # 快速SVD初始化
    )
    
    print("PiSSA配置:")
    print(f"  rank (r): {pissa_config.r}")
    print(f"  alpha: {pissa_config.lora_alpha}")
    print(f"  dropout: {pissa_config.lora_dropout}")
    print(f"  初始化方法: {pissa_config.init_lora_weights}")
    print(f"  scaling factor: {pissa_config.lora_alpha / pissa_config.r}")
    print(f"  target_modules: {pissa_config.target_modules}")
    

if __name__ == "__main__":
    pissa_example() 