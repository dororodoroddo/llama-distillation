import os
import json
import re

PROMPT_ORIGIN_DIR = "./promptOrigin"
PROMPT_TRAIN_PATH = "./promptTrain/train_data.jsonl"
PROMPT_TEST_PATH = "./promptTest/test_data.jsonl"

def split_conversation_blocks(text: str):
    pattern = r'(The traveler: .*?\"|The mystical llama: .*?\")'
    parts = re.split(pattern, text, flags=re.DOTALL)
    header = parts[0].strip()
    dialogue = [p.strip() for p in parts[1:] if p.strip()]

    blocks = []
    full_prefix = header + "\n"
    for i in range(0, len(dialogue)-1, 2):
        traveler = dialogue[i]
        llama = dialogue[i+1]
        full_prefix += traveler + "\n"
        blocks.append({
            "input": full_prefix.strip(),
            "expected": llama.replace('The mystical llama: ', '').strip()
        })
        full_prefix += llama + "\n"
    return blocks

def generate_train_and_test():
    os.makedirs("./promptTrain", exist_ok=True)
    os.makedirs("./promptTest", exist_ok=True)

    with open(PROMPT_TRAIN_PATH, 'w', encoding='utf-8') as train_file, \
         open(PROMPT_TEST_PATH, 'w', encoding='utf-8') as test_file:

        for filename in os.listdir(PROMPT_ORIGIN_DIR):
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(PROMPT_ORIGIN_DIR, filename)
            with open(path, 'r', encoding='utf-8') as f:
                full_text = f.read().strip()

            # 훈련용 전체 대화 저장
            train_file.write(json.dumps({"text": full_text}, ensure_ascii=False) + "\n")

            # 테스트용 블록 분리
            blocks = split_conversation_blocks(full_text)
            for block in blocks:
                test_file.write(json.dumps(block, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    generate_train_and_test()