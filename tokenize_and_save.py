# tokenize_and_save.py

from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize_and_save():
    print("[1.5] 사전 토크나이징 중...")

    BLOCK_PATH = "./promptTrain/train_data_block.jsonl"
    SAVE_PATH = "./tokenized_dataset"
    TOKENIZER_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 실제 사용 모델로 교체

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    def tokenize_function(example):
        return tokenizer(example["input"], truncation=True)

    dataset = load_dataset("json", data_files=BLOCK_PATH, split="train")

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=32,
        remove_columns=["input", "expected"]
    )

    tokenized_dataset.save_to_disk(SAVE_PATH)
    print(f"[1.5] 토크나이즈 완료 → {SAVE_PATH}")

if __name__ == "__main__":
    tokenize_and_save()
