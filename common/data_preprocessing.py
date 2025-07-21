# 共享数据预处理函数
# 支持TriviaQA和NQ Open数据集的预处理

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union
import json
import os

def load_triviaqa_dataset(split: str = "train", subset: str = "rc.nocontext", max_samples: Optional[int] = None):
    """
    加载TriviaQA数据集 rc子集
    
    Args:
        split: 数据集分割 ("train", "validation", "test")
        subset: 子集选择 ("rc.nocontext", "rc", "unfiltered")
        max_samples: 最大样本数，None表示加载全部
    
    Returns:
        Dataset: 处理后的数据集
    """
    print(f"加载TriviaQA数据集 - split: {split}, subset: {subset}")
    
    # 加载数据集
    dataset = load_dataset("trivia_qa", subset)[split]
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"限制样本数量: {len(dataset)}")
    
    return dataset

def load_natural_questions_dataset(split: str = "train", max_samples: Optional[int] = None):
    """
    加载Natural Questions数据集
    
    Args:
        split: 数据集分割 ("train", "validation")
        max_samples: 最大样本数，None表示加载全部
    
    Returns:
        Dataset: 处理后的数据集
    """
    print(f"加载Natural Questions数据集 - split: {split}")
    
    # 加载数据集
    dataset = load_dataset("natural_questions")[split]
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"限制样本数量: {len(dataset)}")
    
    return dataset

def preprocess_triviaqa(examples, tokenizer, max_length: int = 512):
    """
    预处理TriviaQA数据
    
    Args:
        examples: 批量数据
        tokenizer: tokenizer
        max_length: 最大序列长度
    
    Returns:
        Dict: 处理后的数据
    """
    questions = examples["question"]
    
    # 处理答案 - TriviaQA的答案在answer字段
    if "answer" in examples:
        answers = examples["answer"]
        # 如果答案是字典，提取text字段
        if isinstance(answers[0], dict):
            answer_texts = [ans.get("text", ans.get("value", "")) for ans in answers]
        else:
            answer_texts = answers
    else:
        answer_texts = [""] * len(questions)
    
    # 格式化为问答格式
    texts = []
    for q, a in zip(questions, answer_texts):
        # 确保答案是字符串
        if isinstance(a, list) and len(a) > 0:
            a = a[0]  # 取第一个答案
        elif not isinstance(a, str):
            a = str(a)
            
        text = f"Question: {q}\nAnswer: {a}"
        texts.append(text)
    
    # tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 设置labels为input_ids的副本（用于language modeling）
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def preprocess_natural_questions(examples, tokenizer, max_length: int = 512):
    """
    预处理Natural Questions数据
    
    Args:
        examples: 批量数据
        tokenizer: tokenizer  
        max_length: 最大序列长度
    
    Returns:
        Dict: 处理后的数据
    """
    questions = examples["question"]
    
    # 处理答案 - NQ的答案在annotations字段
    answer_texts = []
    for annotations in examples["annotations"]:
        if annotations and len(annotations) > 0:
            # 提取short_answers或yes_no_answer
            annotation = annotations[0]
            if "short_answers" in annotation and annotation["short_answers"]:
                # 取第一个short answer
                short_answer = annotation["short_answers"][0]
                if "text" in short_answer:
                    answer_texts.append(short_answer["text"])
                else:
                    answer_texts.append("")
            elif "yes_no_answer" in annotation:
                answer_texts.append(str(annotation["yes_no_answer"]))
            else:
                answer_texts.append("")
        else:
            answer_texts.append("")
    
    # 格式化为问答格式
    texts = []
    for q, a in zip(questions, answer_texts):
        text = f"Question: {q}\nAnswer: {a}"
        texts.append(text)
    
    # tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 设置labels为input_ids的副本
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def create_qa_dataset(
    dataset_name: str,
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    创建问答数据集的统一接口
    
    Args:
        dataset_name: 数据集名称 ("triviaqa" 或 "natural_questions")
        tokenizer: tokenizer
        split: 数据集分割
        max_length: 最大序列长度
        max_samples: 最大样本数
    
    Returns:
        Dataset: 处理后的tokenized数据集
    """
    print(f"创建 {dataset_name} 数据集...")
    
    if dataset_name.lower() == "triviaqa":
        # 加载TriviaQA
        raw_dataset = load_triviaqa_dataset(split=split, max_samples=max_samples)
        
        # 预处理
        processed_dataset = raw_dataset.map(
            lambda examples: preprocess_triviaqa(examples, tokenizer, max_length),
            batched=True,
            remove_columns=raw_dataset.column_names
        )
        
    elif dataset_name.lower() == "natural_questions":
        # 加载Natural Questions
        raw_dataset = load_natural_questions_dataset(split=split, max_samples=max_samples)
        
        # 预处理
        processed_dataset = raw_dataset.map(
            lambda examples: preprocess_natural_questions(examples, tokenizer, max_length),
            batched=True,
            remove_columns=raw_dataset.column_names
        )
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    print(f"数据集创建完成，样本数: {len(processed_dataset)}")
    return processed_dataset

def create_sample_qa_dataset(tokenizer, num_samples: int = 100, max_length: int = 512):
    """
    创建示例问答数据集（用于测试）
    
    Args:
        tokenizer: tokenizer
        num_samples: 样本数量
        max_length: 最大序列长度
    
    Returns:
        Dataset: 示例数据集
    """
    print(f"创建示例问答数据集，样本数: {num_samples}")
    
    # 示例问答数据
    sample_qa = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is 2+2?", "4"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What year did World War II end?", "1945"),
        ("What is the smallest country in the world?", "Vatican City"),
        ("Who invented the telephone?", "Alexander Graham Bell"),
        ("What is the speed of light?", "299,792,458 meters per second"),
    ]
    
    # 重复数据以达到指定样本数
    questions = []
    answers = []
    for i in range(num_samples):
        q, a = sample_qa[i % len(sample_qa)]
        questions.append(q)
        answers.append(a)
    
    # 格式化
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]
    
    # tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 创建Dataset
    data_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone()
    }
    
    dataset = Dataset.from_dict(data_dict)
    print(f"示例数据集创建完成，样本数: {len(dataset)}")
    
    return dataset

def save_dataset(dataset: Dataset, save_path: str):
    """保存数据集"""
    dataset.save_to_disk(save_path)
    print(f"数据集已保存到: {save_path}")

def load_saved_dataset(load_path: str) -> Dataset:
    """加载保存的数据集"""
    dataset = Dataset.load_from_disk(load_path)
    print(f"数据集已从 {load_path} 加载，样本数: {len(dataset)}")
    return dataset

# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("="*60)
    print("数据预处理示例")
    print("="*60)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建示例数据集
    sample_dataset = create_sample_qa_dataset(tokenizer, num_samples=20)
    print(f"示例数据集: {sample_dataset}")
    
    # 显示第一个样本
    print(f"\n第一个样本:")
    print(f"Input IDs shape: {sample_dataset[0]['input_ids'].shape}")
    print(f"解码文本: {tokenizer.decode(sample_dataset[0]['input_ids'], skip_special_tokens=True)}")
    
    print("\n实际使用方法:")
    print("1. dataset = create_qa_dataset('triviaqa', tokenizer, max_samples=1000)")
    print("2. dataset = create_qa_dataset('natural_questions', tokenizer, max_samples=1000)")
    print("3. save_dataset(dataset, 'path/to/save')")
    print("4. dataset = load_saved_dataset('path/to/save')") 