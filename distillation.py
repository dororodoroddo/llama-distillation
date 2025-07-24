# distillation.py
# TinyLlama 모델 학습 및 GGUF 변환까지 수행합니다.

import os
import json
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from utils.download_model import ensure_model_exists



with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config["student_model"]
MODEL_PATH = ensure_model_exists(MODEL_NAME)
TRAIN_FILE = config["train_file"]
OUTPUT_DIR = config["output_dir"]
GGUF_PATH = config["gguf_output"]
MAX_LEN = config["max_length"]
BATCH = config["batch_size"]
EPOCHS = config["epochs"]
LR = config["learning_rate"]

def train_and_convert():
    print("[distillation] 학습 시작")

    # 학습 데이터 로드
    dataset = load_dataset("json", data_files=TRAIN_FILE)["train"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # 텍스트 → input_ids + labels 변환
    def tokenize(example):
        tokenized = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize)

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

    # 학습 설정
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch",
        fp16=config.get("fp16", False),
        report_to="none"
    )

    # 학습 시작
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # GGUF 변환
    print("[distillation] GGUF 변환 중...")
    convert_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "convert.py"))
    subprocess.run([
        "python3", convert_script,
        OUTPUT_DIR,
        "--outfile", GGUF_PATH,
        "--vocab-type", "spm"
    ])
    print(f"[distillation] GGUF 변환 완료 → {GGUF_PATH}")

if __name__ == "__main__":
    train_and_convert()
