import os
import torch
import json
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tokenize_dataset(tokenizer, dataset, max_length):
    def tokenize_function(example_batch):
        if example_batch.get("text") is not None:
            try:

                a = tokenizer(
                     [example_batch["text"][0]],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
                )
                return a
            except BaseException as e:
                raise RuntimeError("문제가 발생했습니다!")
        elif example_batch.get("input") is not None and example_batch.get("expected") is not None:
            prompts = [i + "\n" + e for i, e in zip(example_batch["input"], example_batch["expected"])]
            return tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        else:
            raise ValueError("Unknown format for example_batch")

    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

def train_and_convert():
    print("[distillation] 학습 시작")

    config = load_config("config.json")
    base_model_path = config["base_model"]
    train_path_full = config["train_data_path_full"]
    train_path_block = config["train_data_path_block"]
    output_dir = config["output_dir"]
    max_length = config.get("max_length", 1024)
    fp16 = config.get("fp16", True)
    batch_size = config.get("batch_size", 1)
    epochs = config.get("epochs", 3)
    gguf_name = config["gguf_name"]

    # CUDA 상태 초기화 (메모리 누수 방지)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("[✓] CUDA 캐시 초기화 및 메모리 통계 리셋")

    # 모델 및 토크나이저 불러오기
    print("[✓] 모델 불러오는 중:", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # 학습 데이터셋 로드 및 결합
    dataset_full = load_dataset("json", data_files=train_path_full)["train"]
    dataset_block = load_dataset("json", data_files=train_path_block)["train"]
    print(f"[✓] Full 예제 수: {len(dataset_full)}, Block 예제 수: {len(dataset_block)}")

    dataset = concatenate_datasets([dataset_full, dataset_block])

    # 토크나이징 (메모리 관리 고려)
    print("[•] 데이터 토크나이징 중...")
    tokenized_dataset = tokenize_dataset(tokenizer, dataset, max_length)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        prediction_loss_only=True,
        logging_steps=10,
        save_strategy="no",
        fp16=fp16 and torch.cuda.is_available(),
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 학습 실행
    print("[→] 학습 시작...")
    trainer.train()
    print("[✓] 학습 완료")

    if torch.cuda.is_available():
        print(f"[GPU 메모리 사용량] {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        torch.cuda.empty_cache()

    # 모델 저장
    print("[→] 모델 저장 중...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # GGUF 변환
    try:
        print("[→] GGUF 변환 시작...")
        os.system(f"python ../llama.cpp/convert.py --outfile {gguf_name} --outdir ./ouputGguf {output_dir}")
        print("[✓] GGUF 변환 완료")
    except Exception as e:
        print("[!] GGUF 변환 실패:", e)

if __name__ == "__main__":
    train_and_convert()
