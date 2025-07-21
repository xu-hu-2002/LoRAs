# AdaLoRA (Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning) Configuration
# AdaLoRA通过自适应预算分配实现动态秩调整

from dataclasses import dataclass, field
from typing import Optional, Union, List
from peft import LoraConfig

@dataclass 
class AdaLoRAConfig:
    """
    AdaLoRA (Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning) 配置类
    
    AdaLoRA的核心特点：
    - 使用SVD分解的适配矩阵
    - 动态秩分配基于重要性分数
    - 自适应预算调度
    - 正交正则化约束
    
    Args:
        r: 初始LoRA秩
        target_rank: 目标平均秋
        init_warmup: 初始预热步数
        final_warmup: 最终预热步数
        mask_interval: 掩码间隔
        beta1: 敏感性EMA系数
        beta2: 不确定性EMA系数
        orth_reg_weight: 正交正则化权重
    """
    
    # 基本LoRA参数
    r: int = field(default=12, metadata={"help": "初始LoRA秩"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA scaling参数"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout概率"})
    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "目标模块列表"})
    
    # AdaLoRA特有参数
    target_rank: int = field(default=8, metadata={"help": "目标平均秩"})
    init_warmup: int = field(default=500, metadata={"help": "初始预热步数"})
    final_warmup: int = field(default=1500, metadata={"help": "最终预热步数"})
    mask_interval: int = field(default=10, metadata={"help": "掩码间隔"})
    
    # EMA参数
    beta1: float = field(default=0.85, metadata={"help": "敏感性平滑的EMA系数"})
    beta2: float = field(default=0.85, metadata={"help": "不确定性量化的EMA系数"})
    
    # 正交正则化
    orth_reg_weight: float = field(default=0.1, metadata={"help": "正交正则化权重"})
    
    # 训练参数
    total_step: Optional[int] = field(default=None, metadata={"help": "总训练步数"})
    target_total_rank: Optional[int] = field(default=None, metadata={"help": "指定的最终总秩"})

def create_adalora_config(**kwargs):
    """
    创建AdaLoRA配置
    
    Example:
        >>> config = create_adalora_config(
        ...     r=12,
        ...     target_rank=8,
        ...     target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ...     init_warmup=500,
        ...     final_warmup=1500,
        ...     mask_interval=10,
        ...     beta1=0.85,
        ...     beta2=0.85
        ... )
    """
    return AdaLoRAConfig(**kwargs)

# AdaLoRA使用示例
ADALORA_USAGE_EXAMPLE = """
# AdaLoRA完整使用示例

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from adalora import SVDLinear, RankAllocator, compute_orth_regu
import loralib

# 1. 模型适配
class AdaptedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # 将目标层替换为SVDLinear
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in 
                ["q_proj", "k_proj", "v_proj", "o_proj"]):
                # 替换为SVDLinear
                new_layer = SVDLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=12,  # 初始秩
                    lora_alpha=32,
                    lora_dropout=0.1
                )
                # 复制原始权重
                new_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_layer.bias.data = module.bias.data.clone()
                
                # 替换模块
                parent = self._get_parent_module(name)
                setattr(parent, name.split('.')[-1], new_layer)

# 2. 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

# 3. 应用AdaLoRA适配
model = AdaptedModel(model)

# 4. 标记只有LoRA参数可训练
loralib.mark_only_lora_as_trainable(model)

# 5. 创建RankAllocator
rankallocator = RankAllocator(
    model, 
    lora_r=12,           # 初始秩
    target_rank=8,       # 目标秋
    init_warmup=500,     # 初始预热
    final_warmup=1500,   # 最终预热
    mask_interval=10,    # 掩码间隔
    total_step=3000,     # 总步数
    beta1=0.85,          # EMA系数1
    beta2=0.85,          # EMA系数2
)

# 6. 准备训练
def training_step(model, batch, optimizer, rankallocator, global_step):
    # 前向传播
    outputs = model(**batch)
    loss = outputs.loss
    
    # 添加正交正则化
    orth_loss = compute_orth_regu(model, regu_weight=0.1)
    total_loss = loss + orth_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # AdaLoRA秩分配
    curr_rank, mask_threshold = rankallocator.update_and_mask(model, global_step)
    
    return total_loss.item(), curr_rank

# 7. 训练循环
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
global_step = 0

for epoch in range(num_epochs):
    for batch in dataloader:
        loss, curr_rank = training_step(
            model, batch, optimizer, rankallocator, global_step
        )
        
        if global_step % 100 == 0:
            print(f"Step {global_step}, Loss: {loss:.4f}, Current Rank: {curr_rank}")
            
        global_step += 1

# 8. 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'rank_pattern': rankallocator.get_rank_pattern(),
}, 'adalora_checkpoint.pt')

print("AdaLoRA训练完成！")
"""

# AdaLoRA vs LoRA比较
ADALORA_VS_LORA = """
AdaLoRA vs LoRA比较：

1. 参数分配：
   - LoRA: 固定秩，均匀分配参数
   - AdaLoRA: 动态秩，重要性驱动的参数分配

2. 适配方式：
   - LoRA: W = W₀ + α·B·A
   - AdaLoRA: W = W₀ + α·B·E·A (E是动态对角矩阵)

3. 参数效率：
   - LoRA: 固定参数量
   - AdaLoRA: 可变参数量，更高效利用预算

4. 训练复杂度：
   - LoRA: 简单，直接训练
   - AdaLoRA: 复杂，需要重要性分数计算和秩分配

5. 性能：
   - LoRA: 良好的基础性能
   - AdaLoRA: 通常优于LoRA，特别是在参数预算有限时

6. 特殊技术：
   - AdaLoRA使用正交正则化
   - 敏感性和不确定性量化
   - 自适应预算调度

7. 适用场景：
   - LoRA: 简单快速的参数高效微调
   - AdaLoRA: 需要最佳参数效率的场景
"""

# AdaLoRA最佳实践
ADALORA_BEST_PRACTICES = """
AdaLoRA最佳实践：

1. 超参数设置：
   - 初始秋 r: 12-16 (略高于目标秩)
   - 目标秩 target_rank: 8-12
   - beta1, beta2: 0.8-0.9 (EMA系数)
   - 正交正则化权重: 0.1

2. 预热策略：
   - init_warmup: 总步数的10-20%
   - final_warmup: 总步数的20-30%
   - mask_interval: 10-50步

3. 模型适配：
   - 主要关注注意力模块和MLP
   - 避免对输出层应用AdaLoRA
   - 考虑层间的重要性差异

4. 训练技巧：
   - 使用较小的学习率 (1e-4 to 5e-4)
   - 监控秩分配模式
   - 观察正交正则化损失

5. 调试技巧：
   - 检查重要性分数的分布
   - 监控mask_threshold的变化
   - 验证秩分配是否合理

6. 性能优化：
   - 在GPU上运行重要性分数计算
   - 考虑梯度检查点以节省内存
   - 适当调整mask_interval以平衡性能和计算成本

7. 评估指标：
   - 参数效率 (性能/参数量)
   - 秋利用率
   - 收敛速度
"""

if __name__ == "__main__":
    print("AdaLoRA Configuration Example:")
    config = create_adalora_config(r=12, target_rank=8)
    print(config)
    
    print("\nUsage Example:")
    print(ADALORA_USAGE_EXAMPLE)
    
    print("\nAdaLoRA vs LoRA:")
    print(ADALORA_VS_LORA)
    
    print("\nBest Practices:")
    print(ADALORA_BEST_PRACTICES) 