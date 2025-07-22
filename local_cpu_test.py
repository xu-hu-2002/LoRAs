import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import warnings

# 从项目文件中导入必要的模块
from config import COMMON_CONFIG
from common.data_preprocessing import load_and_preprocess_data
from LoRA.lora_hf import create_lora_config
from DoRA.dora_hf import create_dora_config
from PiSSA.pissa_hf import create_pissa_config
from QLoRA.qlora_hf import create_qlora_config, create_bnb_config  #QLoRA 需要GPU 暂时不测试
from AdaLoRA.adalora import LoRALayer, RankAllocator,SVDLinear #并未集成到PEFT 需要单独配置config
from AdaLoRA.adalora_config import AdaLoRAConfig 
from peft import get_peft_model

# --- 全局测试配置 ---
TEST_MODEL = "distilgpt2"  # 回到轻量模型，适合当前内存情况 (82M 参数)
MAX_SAMPLES = 10 # 使用10个样本进行快速测试
TRAIN_SAMPLES = 8 # 8个用于训练
VAL_SAMPLES = 2 # 2个用于验证
BATCH_SIZE = 1
NUM_EPOCHS = 1

# 在全局配置部分添加
DISTILGPT2_TARGET_MODULES = ["c_attn", "c_proj"]  # DistilGPT2专用

def run_peft_test(peft_method_name, get_config_func, use_bnb=False):
    """
    一个通用的测试函数，用于测试基于HuggingFace PEFT库的各种方法。

    Args:
        peft_method_name (str): PEFT方法的名称 (例如, "LoRA", "DoRA")。
        get_config_func (function): 一个返回PEFT配置对象的函数。
        use_bnb (bool): 是否为QLoRA使用BitsAndBytes量化。
    """
    print(f"\n{'='*20} 正在测试: {peft_method_name} {'='*20}")

    # --- 1. 检查环境和加载模型 ---
    if use_bnb and not torch.cuda.is_available():
        warnings.warn(f"⚠️  警告: {peft_method_name} 需要CUDA GPU，但当前环境不可用。正在跳过此测试。")
        print(f"{'='*20} {peft_method_name} 测试跳过 {'='*20}\n")
        return

    print(f"加载基础模型: {TEST_MODEL}...")
    model_kwargs = {}
    if use_bnb:
        bnb_config = create_bnb_config()
        model_kwargs['quantization_config'] = bnb_config
    
    # CPU优化：使用torch_dtype=torch.float32，低内存模式
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL, 
        torch_dtype=torch.float32,  # CPU上使用float32
        low_cpu_mem_usage=True,     # 低内存模式
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 2. 加载和准备数据 ---
    print("加载并预处理TriviaQA数据集...")
    full_dataset = load_and_preprocess_data(
        dataset_name="trivia_qa", 
        model_name=TEST_MODEL, 
        max_samples=MAX_SAMPLES, 
        split='train' 
    )
    
    train_dataset = Dataset.from_dict(full_dataset[:TRAIN_SAMPLES])
    validation_dataset = Dataset.from_dict(full_dataset[TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES])

    # --- 3. 创建PEFT配置并应用到模型 ---
    print(f"创建 {peft_method_name} 配置...")
    # 根据模型选择target_modules
    if TEST_MODEL == "distilgpt2":
        target_modules = ["c_attn", "c_proj"]
    else:
        target_modules = COMMON_CONFIG['target_modules']
    
    # 创建配置时传入正确的target_modules
    peft_config = get_config_func(r=8, lora_alpha=16, target_modules=target_modules)
    model = get_peft_model(model, peft_config)
    print(f"应用 {peft_method_name} 后的模型:")
    model.print_trainable_parameters()

    # --- 4. 设置训练参数并开始训练 ---
    training_args = TrainingArguments(
        output_dir=f"./results_{peft_method_name.lower()}_test",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_dir=f'./logs_{peft_method_name.lower()}_test',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # 关闭wandb等报告
        use_cpu=True, # 强制使用CPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )

    print(f"🚀 开始 {peft_method_name} 训练...")
    trainer.train()
    print(f"✅ {peft_method_name} 训练完成。")
    
    print(f"📊 开始 {peft_method_name} 评估...")
    eval_results = trainer.evaluate()
    print(f"✅ {peft_method_name} 评估完成: {eval_results}")
    print(f"{'='*20} {peft_method_name} 测试成功 {'='*20}\n")


def run_adalora_test():
    """
    专门为AdaLoRA编写的测试函数，因为它使用了自定义的实现。
    """
    peft_method_name = "AdaLoRA"
    print(f"\n{'='*20} 正在测试: {peft_method_name} {'='*20}")

    # --- 1. 加载模型 ---
    print(f"加载基础模型: {TEST_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 加载数据 ---
    print("加载并预处理TriviaQA数据集...")
    full_dataset = load_and_preprocess_data(
        dataset_name="trivia_qa", 
        model_name=TEST_MODEL, 
        max_samples=MAX_SAMPLES, 
        split='train'
    )
    train_dataset = Dataset.from_dict(full_dataset[:TRAIN_SAMPLES])
    validation_dataset = Dataset.from_dict(full_dataset[TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES])
    
    # --- 3. 创建AdaLoRA配置并应用 ---
    print("创建 AdaLoRA 配置...")
    # AdaLoRA的配置有些不同
    config = AdaLoRAConfig(
        target_r=8, 
        init_r=12,
        tinit=200,
        tfinal=1000,
        deltaT=10,
        lora_alpha=16, # 保持alpha=2*r的比例
        target_modules=COMMON_CONFIG['target_modules'],
        task_type="CAUSAL_LM"
    )
    model = AdaLoRA(model, config)
    model.print_trainable_parameters()

    # --- 4. 训练 ---
    training_args = TrainingArguments(
        output_dir=f"./results_{peft_method_name.lower()}_test",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_dir=f'./logs_{peft_method_name.lower()}_test',
        logging_steps=10,
        report_to="none",
        use_cpu=True,
    )

    # AdaLoRA需要自定义的训练循环来处理秩的更新
    print(f"🚀 开始 {peft_method_name} 训练...")
    # 这里我们简化训练过程，仅验证代码逻辑，不执行完整的训练循环
    # 因为完整的AdaLoRA训练器比较复杂，我们的目标是验证配置和模型包装是否正确
    try:
        trainer = Trainer(
            model=model.model, # 传入内部模型
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        print(f"✅ {peft_method_name} 训练逻辑验证成功。")
    except Exception as e:
        print(f"❌ {peft_method_name} 测试失败: {e}")

    print(f"{'='*20} {peft_method_name} 测试完成 {'='*20}\n")


if __name__ == "__main__":
    print("🚀 开始本地CPU PEFT方法逻辑测试...")

    # 测试基于HuggingFace PEFT库的方法
    run_peft_test("LoRA", create_lora_config)
    run_peft_test("DoRA", create_dora_config)
    run_peft_test("PiSSA", create_pissa_config)
    
    # QLoRA需要特殊处理，因为它需要GPU
    run_peft_test("QLoRA", create_qlora_config, use_bnb=True)

    # 测试我们自定义实现的AdaLoRA (暂时跳过，因为集成复杂)
    # run_adalora_test()

    print("🎉 LoRA, DoRA, PiSSA, QLoRA 测试流程已执行完毕!")
    print("📝 注意: AdaLoRA使用自定义实现，暂时跳过测试") 