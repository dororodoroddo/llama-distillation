import os
import json
import subprocess
import torch
import shutil
from datasets import load_from_disk
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class LoggingDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        print(f"[🧾] Using sample index: {idx}")
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class SavePermanentCheckpointCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 200 == 0:
            src_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            dst_dir = os.path.join(args.output_dir, f"../save-point/checkpoint-{state.global_step}-permanent")
            if os.path.exists(src_dir) and not os.path.exists(dst_dir):
                shutil.copytree(src_dir, dst_dir)
                print(f"[✓] Permanent checkpoint saved at {dst_dir}")


def convert_to_gguf(gguf_name, output_dir):
    try:
        print("[→] GGUF 변환 시작...")
        subprocess.run([
            "python", "../llama.cpp/convert.py",
            "--outfile", gguf_name,
            "--outdir", "./ouputGguf",
            output_dir
        ], check=True)
        print("[✓] GGUF 변환 완료")
    except subprocess.CalledProcessError as e:
        print(f"[!] GGUF 변환 실패: {e}")


def train_and_convert():
    print("[CPU 기반 학습 시작]")

    config = load_config("config.json")
    base_model_path = config["base_model"]
    output_dir = config["output_dir"]
    gguf_name = config["gguf_name"]
    batch_size = config.get("batch_size", 1)
    epochs = config.get("epochs", 3)
    set_seed(77)

    print("[✓] 모델 및 토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32
    ).to("cpu")  # 명시적 CPU 로딩

    print("[✓] 토크나이즈된 데이터셋 로딩 중: ./tokenized_dataset")
    dataset = LoggingDataset(load_from_disk("./tokenized_dataset").shuffle(seed=77))

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=1,
        prediction_loss_only=True,
        logging_steps=1,
        save_strategy="steps",           # ← 저장 전략을 "스텝 단위"로 변경
        save_steps=4,                    # ← 4 스텝마다 저장
        save_total_limit=2,             # ← 최근 2개만 유지 (디스크 공간 절약)
        fp16=False,                     # CPU 전용
        report_to="none",
        dataloader_num_workers=0        # CPU 병렬 처리 최소화
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SavePermanentCheckpointCallback()]
    )

    print("[→] 학습 시작...")
    output_dir = "outputs/distilled"
    checkpoint_exists = (
        os.path.isdir(output_dir)
        and any("checkpoint" in d for d in os.listdir(output_dir))
    )

    if checkpoint_exists:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    print("[✓] 학습 완료")

    print("[→] 모델 저장 중...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    convert_to_gguf(gguf_name, output_dir)


if __name__ == "__main__":
    train_and_convert()
