# LoRA (Low-Rank Adaptation) - HuggingFace PEFT实现
# 使用HuggingFace PEFT库，只需配置即可

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
sys.path.append('..')
from config import get_model, get_peft_config, COMMON_CONFIG

def create_lora_config(size="medium", **kwargs):
    """
    创建LoRA配置，从config文件获取推荐参数
    
    Args:
        size: 模型大小 ("small", "medium", "large", "xl")
        **kwargs: 覆盖默认配置的参数
    """
    # 获取推荐配置
    peft_config = get_peft_config(size)
    
    # 合并配置
    config = {
        "r": peft_config["r"],
        "lora_alpha": peft_config["alpha"],
        **COMMON_CONFIG,
        **kwargs  # 用户自定义参数会覆盖默认值
    }
    
    print(f"LoRA配置 (size={size}): r={config['r']}, alpha={config['lora_alpha']}")
    
    return LoraConfig(**config)

def load_model_with_lora(
    model_name: str = None,
    lora_config: LoraConfig = None,
    size: str = "medium",
    torch_dtype=torch.bfloat16,
    device_map="auto"
):
    """
    加载模型并应用LoRA，从config获取推荐模型
    
    Args:
        model_name: 模型名称，如果为None则从config获取
        lora_config: LoRA配置，如果为None则使用默认配置
        size: 模型大小，用于获取推荐模型和配置
        torch_dtype: 模型数据类型
        device_map: 设备映射
    
    Returns:
        tuple: (model, tokenizer)
    """
    # 从config获取模型名称
    if model_name is None:
        model_name = get_model(size)
    
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
    
    # 应用LoRA
    if lora_config is None:
        lora_config = create_lora_config(size)
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def save_lora_adapter(model, save_path: str):
    """保存LoRA适配器"""
    model.save_pretrained(save_path)
    print(f"LoRA适配器已保存到: {save_path}")

def load_lora_adapter(base_model_name: str, adapter_path: str):
    """加载LoRA适配器"""
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model

# 使用示例
def lora_example():
    """LoRA完整使用示例"""
    print("="*60)
    print("LoRA (HuggingFace PEFT) 使用示例")
    print("="*60)
    
    # 1. 创建LoRA配置
    # 使用配置文件的推荐设置
    print("推荐配置:")
    print(f"  模型: {get_model('medium')}")
    
    lora_config = create_lora_config("medium")
    
    # 实际使用示例（注释掉以避免真正下载）
    # model, tokenizer = load_model_with_lora(size="medium")
    
    print("\n使用方法:")
    print("1. model, tokenizer = load_model_with_lora()  # 默认medium模型")
    print("2. model, tokenizer = load_model_with_lora(size='large')  # 8B模型")
    print("3. 训练模型...")
    print("4. save_lora_adapter(model, 'path/to/adapter')")

if __name__ == "__main__":
    lora_example() 