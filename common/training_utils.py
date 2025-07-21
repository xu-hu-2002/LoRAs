# 共享训练工具函数
# 提供统一的训练接口和工具

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel
from datasets import Dataset
from typing import Optional, Dict, Any
import os
from datetime import datetime

def create_training_arguments(
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    evaluation_strategy: str = "no",
    save_strategy: str = "steps",
    fp16: bool = True,
    bf16: bool = False,
    max_grad_norm: float = 1.0,
    dataloader_num_workers: int = 0,
    remove_unused_columns: bool = False,
    **kwargs
) -> TrainingArguments:
    """
    创建训练参数
    
    Args:
        output_dir: 输出目录
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 每设备batch size
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率（PEFT推荐2e-4）
        warmup_steps: 预热步数
        logging_steps: 日志记录步数
        save_steps: 模型保存步数
        eval_steps: 评估步数
        evaluation_strategy: 评估策略
        save_strategy: 保存策略
        fp16: 是否使用FP16
        bf16: 是否使用BF16
        max_grad_norm: 梯度裁剪
        dataloader_num_workers: 数据加载工作进程数
        remove_unused_columns: 是否移除未使用的列
        **kwargs: 其他参数
    
    Returns:
        TrainingArguments: 训练参数
    """
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=remove_unused_columns,
        report_to=[],  # 禁用wandb等报告
        save_total_limit=2,  # 只保留最近2个checkpoint
        load_best_model_at_end=False,
        **kwargs
    )
    
    print(f"训练参数:")
    print(f"  输出目录: {output_dir}")
    print(f"  训练轮数: {num_train_epochs}")
    print(f"  Batch size: {per_device_train_batch_size}")
    print(f"  梯度累积: {gradient_accumulation_steps}")
    print(f"  学习率: {learning_rate}")
    print(f"  混合精度: FP16={fp16}, BF16={bf16}")
    
    return training_args

def create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    training_args: Optional[TrainingArguments] = None,
    data_collator = None
) -> Trainer:
    """
    创建训练器
    
    Args:
        model: 模型
        tokenizer: tokenizer
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        training_args: 训练参数
        data_collator: 数据整理器
    
    Returns:
        Trainer: 训练器
    """
    
    if training_args is None:
        training_args = create_training_arguments()
    
    if data_collator is None:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # 对于causal LM，不使用MLM
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print(f"训练器创建完成")
    print(f"  训练样本数: {len(train_dataset)}")
    if eval_dataset:
        print(f"  验证样本数: {len(eval_dataset)}")
    
    return trainer

def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    training_args: Optional[TrainingArguments] = None,
    resume_from_checkpoint: Optional[str] = None
) -> Trainer:
    """
    训练模型的完整流程
    
    Args:
        model: 模型
        tokenizer: tokenizer
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        training_args: 训练参数
        resume_from_checkpoint: 从checkpoint恢复训练
    
    Returns:
        Trainer: 训练完成的训练器
    """
    
    print("="*60)
    print("开始模型训练")
    print("="*60)
    
    # 打印模型信息
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    
    # 创建训练器
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args
    )
    
    # 开始训练
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("训练完成！")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise
    
    return trainer

def save_model_and_adapter(
    model,
    tokenizer,
    save_dir: str,
    save_adapter_only: bool = True
):
    """
    保存模型和适配器
    
    Args:
        model: 模型
        tokenizer: tokenizer
        save_dir: 保存目录
        save_adapter_only: 是否只保存适配器
    """
    
    print(f"保存模型到: {save_dir}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存tokenizer
    tokenizer.save_pretrained(save_dir)
    print("Tokenizer已保存")
    
    if save_adapter_only and isinstance(model, PeftModel):
        # 只保存PEFT适配器
        model.save_pretrained(save_dir)
        print("PEFT适配器已保存")
    else:
        # 保存完整模型
        model.save_pretrained(save_dir)
        print("完整模型已保存")

def evaluate_model(
    model,
    tokenizer,
    eval_dataset: Dataset,
    batch_size: int = 8
) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: 模型
        tokenizer: tokenizer
        eval_dataset: 评估数据集
        batch_size: batch size
    
    Returns:
        Dict[str, float]: 评估指标
    """
    
    print("开始模型评估...")
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    metrics = {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity
    }
    
    print(f"评估完成:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")
    
    return metrics

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    """
    生成文本
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入提示
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        do_sample: 是否采样
        top_p: nucleus sampling参数
        top_k: top-k sampling参数
    
    Returns:
        str: 生成的文本
    """
    
    model.eval()
    
    # tokenize输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除输入部分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

# 使用示例
if __name__ == "__main__":
    print("="*60)
    print("训练工具函数示例")
    print("="*60)
    
    # 创建训练参数示例
    training_args = create_training_arguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=2e-4
    )
    
    print(f"\n训练参数示例: {training_args.output_dir}")
    
    print("\n使用方法:")
    print("1. training_args = create_training_arguments()")
    print("2. trainer = train_model(model, tokenizer, train_dataset, training_args=training_args)")
    print("3. save_model_and_adapter(model, tokenizer, './saved_model')")
    print("4. metrics = evaluate_model(model, tokenizer, eval_dataset)")
    print("5. text = generate_text(model, tokenizer, 'Question: What is AI?')") 