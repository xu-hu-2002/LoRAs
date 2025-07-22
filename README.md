# PEFT Collection

A unified implementation of five mainstream Parameter-Efficient Fine-Tuning (PEFT) techniques based on HuggingFace PEFT library, optimized for TriviaQA and Natural Questions (NQ Open) datasets.

## 🌟 Features

- **5 PEFT Methods**: LoRA, DoRA, QLoRA, PiSSA, AdaLoRA
- **Unified Interface**: Consistent API across all methods
- **Dataset Support**: TriviaQA and Natural Questions
- **Ready-to-Use**: Complete examples and configurations
- **Memory Efficient**: Optimized for various hardware constraints

## 📁 Project Structure

```
LoRA/
├── LoRA/
│   └── lora_hf.py          # LoRA - HuggingFace PEFT Implementation
├── DoRA/
│   └── dora_hf.py          # DoRA - HuggingFace PEFT Implementation
├── QLoRA/
│   └── qlora_hf.py         # QLoRA - HuggingFace PEFT Implementation
├── PiSSA/
│   └── pissa_hf.py         # PiSSA - HuggingFace PEFT Implementation
├── AdaLoRA/
│   ├── adalora.py          # AdaLoRA Core Implementation (Standalone)
│   └── adalora_config.py   # AdaLoRA Configuration
├── common/
│   ├── data_preprocessing.py  # Shared Data Preprocessing
│   └── training_utils.py      # Shared Training Utilities
├── example_usage.py        # Complete Usage Examples
├── local_cpu_test.py       # Local CPU Testing
├── test_adalora.py         # AdaLoRA Testing
├── config.py               # Global Configuration
├── requirements.txt        # Dependencies
└── README.md              # This Document
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Demo

```bash
python example_usage.py
```

### 3. Basic Usage

```python
# Example with LoRA
from LoRA.lora_hf import create_lora_config, load_model_with_lora
from common.data_preprocessing import create_qa_dataset
from common.training_utils import train_model

# 1. Create configuration
config = create_lora_config(r=128, lora_alpha=256) 

# 2. Load model
model, tokenizer = load_model_with_lora(size="medium")  # Uses Llama-3.2-3B

# 3. Prepare data
dataset = create_qa_dataset("triviaqa", tokenizer, max_samples=1000) 

# 4. Train
trainer = train_model(model, tokenizer, dataset)
```

## 🔧 Supported Methods

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
    use_dora=True,           # Key: Enable DoRA
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
)
model, tokenizer = load_model_with_dora("model_name", config)
```

### 3. QLoRA (4-bit Quantized LoRA)
```python
from QLoRA.qlora_hf import create_qlora_config, load_model_with_qlora, create_bnb_config

# Quantization configuration
bnb_config = create_bnb_config(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA configuration
qlora_config = create_qlora_config(r=128, lora_alpha=256)

model, tokenizer = load_model_with_qlora("model_name", qlora_config, bnb_config)
```

### 4. PiSSA (Principal Singular Values and Singular Vectors Adaptation)
```python
from PiSSA.pissa_hf import create_pissa_config, load_model_with_pissa

config = create_pissa_config(
    r=128,
    lora_alpha=256,
    lora_dropout=0.0,        # PiSSA recommends 0
    init_lora_weights="pissa_niter_4"  # SVD initialization
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
# Note: AdaLoRA requires special training loop
```

## 📊 Data Processing

### Supported Datasets
- **TriviaQA**: Open-domain question answering
- **Natural Questions (NQ Open)**: Real-world Q&A from Google search

### Data Preprocessing
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

# Sample data (for testing)
sample_dataset = create_sample_qa_dataset(tokenizer, num_samples=10)
```

## 🏋️ Training Configuration

### Default Training Parameters
```python
from common.training_utils import create_training_arguments, train_model

training_args = create_training_arguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,      # PEFT recommended learning rate
    warmup_steps=100,
    fp16=True,               # Mixed precision training
    logging_steps=10,
    save_steps=500
)

trainer = train_model(model, tokenizer, dataset, training_args=training_args)
```

## 💡 Method Selection Guide （to be decided)

| Method | Use Case | Memory | Performance | Complexity |
|--------|----------|---------|-------------|------------|
| **LoRA** | Quick experiments | Medium | Good | Low |
| **DoRA** | Performance priority | Medium | Better | Medium |
| **QLoRA** | Memory constrained | Low | Good | Medium |
| **PiSSA** | Best efficiency | Medium | Best | Medium |
| **AdaLoRA** | Adaptive parameters | Medium | Good | High |

**Recommendations:** (to be decided)
- **Quick prototyping**: LoRA
- **Best performance**: DoRA or PiSSA  
- **Limited GPU memory**: QLoRA
- **Parameter efficiency**: AdaLoRA
- **Production use**: PiSSA + QLoRA combination

## 🔧 Advanced Usage

### Save and Load Models
```python
from common.training_utils import save_model_and_adapter

# Save adapter (recommended)
save_model_and_adapter(model, tokenizer, "./my_adapter", save_adapter_only=True)

# Load adapter
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base_model_name")
model = PeftModel.from_pretrained(base_model, "./my_adapter")
```

### Model Evaluation
```python
from common.training_utils import evaluate_model, generate_text

# Evaluation
metrics = evaluate_model(model, tokenizer, eval_dataset)
print(f"Loss: {metrics['eval_loss']:.4f}")

# Text generation
response = generate_text(
    model, tokenizer, 
    "Question: What is the capital of France?",
    max_new_tokens=50
)
print(response)
```

### Local Testing
```python
# Run local CPU tests
python local_cpu_test.py

# Test specific method
python test_adalora.py
```

## 🛠️ Configuration

The `config.py` file contains global settings:
- Model configurations
- Training hyperparameters  
- Dataset paths
- Hardware settings

## 🐛 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Use QLoRA or reduce batch size
2. **Slow training**: Enable mixed precision (`fp16=True`)
3. **Poor convergence**: Adjust learning rate (try 1e-4 to 5e-4)

### Dependencies
Make sure you have compatible versions:
- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `peft >= 0.4.0`
- `datasets >= 2.10.0`

## 📚 References

1. **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **DoRA**: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
3. **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
4. **PiSSA**: [PiSSA: Principal Singular Values and Singular Vectors Adaptation](https://arxiv.org/abs/2404.02948)
5. **AdaLoRA**: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🌟 Acknowledgments

- HuggingFace for the excellent PEFT library
- Original authors of each PEFT method
- The open-source community for inspiration and feedback
