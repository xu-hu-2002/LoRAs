# QLoRA (4-bit Quantized LoRA) - HuggingFace PEFT实现
# QLoRA结合4bit量化和LoRA，大幅减少显存使用

import torch
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

def create_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant: bool = True
):
    """
    创建BitsAndBytes量化配置
    
    Args:
        load_in_4bit: 是否使用4bit量化
        bnb_4bit_quant_type: 量化类型，"nf4"或"fp4"
        bnb_4bit_compute_dtype: 计算时的数据类型
        bnb_4bit_use_double_quant: 是否使用双重量化
    
    Returns:
        BitsAndBytesConfig: 量化配置
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

def create_qlora_config(
    r: int = 128,
    lora_alpha: int = 256,  # 默认 2 * r
    lora_dropout: float = 0.1,
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
):
    """
    创建QLoRA配置
    
    QLoRA核心思想：
    - 基础模型使用4bit量化存储
    - LoRA适配器保持16/32bit精度
    - 使用特殊的反向传播算法
    
    Args:
        r: LoRA rank (秩)，默认128
        lora_alpha: LoRA scaling参数，默认256 (2*r)
        lora_dropout: dropout概率
        target_modules: 目标模块列表
        bias: bias处理方式
        task_type: 任务类型
    
    Returns:
        LoraConfig: PEFT LoRA配置
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
        print(f"QLoRA: 自动设置 lora_alpha = 2 * r = {lora_alpha}")
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type
    )

def load_model_with_qlora(
    model_name: str,
    qlora_config: LoraConfig = None,
    bnb_config: BitsAndBytesConfig = None,
    device_map="auto"
):
    """
    加载模型并应用QLoRA
    
    Args:
        model_name: 模型名称
        qlora_config: QLoRA配置
        bnb_config: BitsAndBytes量化配置
        device_map: 设备映射
    
    Returns:
        model: 应用QLoRA的模型
        tokenizer: tokenizer
    """
    # 创建默认配置
    if bnb_config is None:
        bnb_config = create_bnb_config()
    
    if qlora_config is None:
        qlora_config = create_qlora_config()
    
    # 加载量化模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model)
    
    # 应用LoRA到量化模型
    model = get_peft_model(model, qlora_config)
    model.print_trainable_parameters()
    
    print("QLoRA配置:")
    print(f"  4bit量化: {bnb_config.load_in_4bit}")
    print(f"  量化类型: {bnb_config.bnb_4bit_quant_type}")
    print(f"  双重量化: {bnb_config.bnb_4bit_use_double_quant}")
    print(f"  计算类型: {bnb_config.bnb_4bit_compute_dtype}")
    
    return model, tokenizer

def save_qlora_adapter(model, save_path: str):
    """保存QLoRA适配器"""
    model.save_pretrained(save_path)
    print(f"QLoRA适配器已保存到: {save_path}")

def load_qlora_adapter(base_model_name: str, adapter_path: str, bnb_config: BitsAndBytesConfig = None):
    """加载QLoRA适配器"""
    if bnb_config is None:
        bnb_config = create_bnb_config()
    
    # 加载量化基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 加载QLoRA适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model

# 使用示例
def qlora_example():
    """QLoRA完整使用示例"""
    print("="*60)
    print("QLoRA (HuggingFace PEFT) 使用示例")
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
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1
    )
    
    print("量化配置:")
    print(f"  4bit量化: {bnb_config.load_in_4bit}")
    print(f"  量化类型: {bnb_config.bnb_4bit_quant_type}")
    print(f"  计算类型: {bnb_config.bnb_4bit_compute_dtype}")
    print(f"  双重量化: {bnb_config.bnb_4bit_use_double_quant}")
    
    print("\nLoRA配置:")
    print(f"  rank (r): {qlora_config.r}")
    print(f"  alpha: {qlora_config.lora_alpha}")
    print(f"  dropout: {qlora_config.lora_dropout}")
    print(f"  scaling factor: {qlora_config.lora_alpha / qlora_config.r}")
    print(f"  target_modules: {qlora_config.target_modules}")
    

if __name__ == "__main__":
    qlora_example() 