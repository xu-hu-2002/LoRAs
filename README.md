# 参数高效微调技术集合 (PEFT Collection)

基于HuggingFace PEFT库的五种主流参数高效微调技术统一实现，专为TriviaQA和NQ Open数据集训练优化。


## 📁 项目结构

```
LoRA/
├── LoRA/
│   └── lora_hf.py          # LoRA - HuggingFace PEFT实现
├── DoRA/
│   └── dora_hf.py          # DoRA - HuggingFace PEFT实现
├── QLoRA/
│   └── qlora_hf.py         # QLoRA - HuggingFace PEFT实现
├── PiSSA/
│   └── pissa_hf.py         # PiSSA - HuggingFace PEFT实现
├── AdaLoRA/
│   ├── adalora.py          # AdaLoRA核心实现（独立）
│   └── adalora_config.py   # AdaLoRA配置
├── common/
│   ├── data_preprocessing.py  # 共享数据预处理
│   └── training_utils.py      # 共享训练工具
├── example_usage.py        # 完整使用示例
├── requirements.txt        # 依赖要求
└── README.md              # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python example_usage.py
```

### 3. 基本使用

```python
# 以LoRA为例
from LoRA.lora_hf import create_lora_config, load_model_with_lora
from common.data_preprocessing import create_qa_dataset
from common.training_utils import train_model

# 1. 创建配置
config = create_lora_config(r=128, lora_alpha=256) 

# 2. 加载模型
model, tokenizer = load_model_with_lora(size="medium")  # 使用config推荐的Llama-3.2-3B

# 3. 准备数据
dataset = create_qa_dataset("triviaqa", tokenizer, max_samples=1000) 

# 4. 训练
trainer = train_model(model, tokenizer, dataset)
```

## 🔧 支持的技术

### 1. LoRA (Low-Rank Adaptation)
```python
from LoRA.lora_hf import create_lora_config, load_model_with_lora

config = create_lora_config(
    r=128,                    # rank
    lora_alpha=256,          # 2 * r
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
)
model, tokenizer = load_model_with_lora("model_name", config)
```

### 2. DoRA (Weight-Decomposed Low-Rank Adaptation)
```python
from DoRA.dora_hf import create_dora_config, load_model_with_dora

config = create_dora_config(
    r=128,
    lora_alpha=256,
    use_dora=True,           # 关键：启用DoRA
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
)
model, tokenizer = load_model_with_dora("model_name", config)
```

### 3. QLoRA (4-bit Quantized LoRA)
```python
from QLoRA.qlora_hf import create_qlora_config, load_model_with_qlora, create_bnb_config

# 量化配置
bnb_config = create_bnb_config(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA配置
qlora_config = create_qlora_config(r=128, lora_alpha=256)

model, tokenizer = load_model_with_qlora("model_name", qlora_config, bnb_config)
```

### 4. PiSSA (Principal Singular Values and Singular Vectors Adaptation)
```python
from PiSSA.pissa_hf import create_pissa_config, load_model_with_pissa

config = create_pissa_config(
    r=128,
    lora_alpha=256,
    lora_dropout=0.0,        # PiSSA推荐设为0
    init_lora_weights="pissa_niter_4"  # SVD初始化
)
model, tokenizer = load_model_with_pissa("model_name", config)
```

### 5. AdaLoRA (Adaptive Budget Allocation)
```python
from AdaLoRA.adalora_config import AdaLoRAConfig
from AdaLoRA.adalora import RankAllocator

config = AdaLoRAConfig(
    r=12,
    target_rank=8,
    init_warmup=500,
    final_warmup=1500,
    mask_interval=10
)
# 注意：AdaLoRA需要特殊的训练循环
```

## 📊 数据处理

### 支持的数据集
- **TriviaQA**: 通用知识问答数据集
- **Natural Questions (NQ Open)**: 基于维基百科的开放域问答

### 数据预处理
```python
from common.data_preprocessing import create_qa_dataset

# TriviaQA
trivia_dataset = create_qa_dataset(
    "triviaqa", 
    tokenizer, 
    split="train",
    max_samples=5000
)

# Natural Questions
nq_dataset = create_qa_dataset(
    "natural_questions", 
    tokenizer, 
    split="train", 
    max_samples=5000
)

# 示例数据（用于测试）
sample_dataset = create_sample_qa_dataset(tokenizer, num_samples=10)
```

## 🏋️ 训练配置

### 默认训练参数
```python
from common.training_utils import create_training_arguments, train_model

training_args = create_training_arguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,      # PEFT推荐学习率
    warmup_steps=100,
    fp16=True,               # 混合精度训练
    logging_steps=10,
    save_steps=500
)

trainer = train_model(model, tokenizer, dataset, training_args=training_args)
```


## 💡 选择建议 （待定)

- **快速实验**: LoRA
- **性能优先**: DoRA 或 PiSSA
- **显存受限**: QLoRA
- **参数效率**: AdaLoRA
- **最佳实践**: PiSSA + QLoRA 组合

## 🔧 高级用法

### 保存和加载模型
```python
from common.training_utils import save_model_and_adapter

# 保存适配器（推荐）
save_model_and_adapter(model, tokenizer, "./my_adapter", save_adapter_only=True)

# 加载适配器
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base_model_name")
model = PeftModel.from_pretrained(base_model, "./my_adapter")
```

### 模型评估
```python
from common.training_utils import evaluate_model, generate_text

# 评估
metrics = evaluate_model(model, tokenizer, eval_dataset)
print(f"Loss: {metrics['eval_loss']:.4f}")

# 生成
response = generate_text(
    model, tokenizer, 
    "Question: What is the capital of France?",
    max_new_tokens=50
)
print(response)
```


## 📋 依赖要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- PEFT 0.8+
- BitsAndBytes 0.42+ (用于QLoRA)
- Datasets 2.14+

详见 `requirements.txt`


## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📚 参考文献

1. **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **DoRA**: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
3. **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
4. **PiSSA**: [PiSSA: Principal Singular Values and Singular Vectors Adaptation](https://arxiv.org/abs/2404.02948)
5. **AdaLoRA**: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
