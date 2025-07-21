# 配置文件 - 统一管理模型和PEFT参数
# 简洁版本，包含最常用的配置

# 推荐模型 (按性能和资源需求排序)
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"  # 默认推荐模型

MODELS = {
    "small": "meta-llama/Llama-3.2-1B",     # 1.24B参数，适合快速测试
    "medium": "meta-llama/Llama-3.2-3B",    # 3.21B参数，性价比最高 ⭐推荐⭐
    "large": "meta-llama/Llama-3.1-8B",     # 8.03B参数，高性能
    "xl": "meta-llama/Llama-3.1-70B",       # 70.6B参数，最佳性能
}

# 基础PEFT配置
PEFT_CONFIG = {
    "small": {"r": 32, "alpha": 64},         # 1B模型推荐
    "medium": {"r": 128, "alpha": 256},       # 3B模型推荐 ⭐
    "large": {"r": 128, "alpha": 256},       # 8B模型推荐
    "xl": {"r": 256, "alpha": 512},          # 70B模型推荐
}

# 通用参数
COMMON_CONFIG = {
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# 获取模型配置的便捷函数
def get_model(size="medium"):
    """获取模型名称"""
    return MODELS.get(size, DEFAULT_MODEL)

def get_peft_config(size="medium"):
    """获取PEFT配置"""
    return PEFT_CONFIG.get(size, PEFT_CONFIG["medium"]) 