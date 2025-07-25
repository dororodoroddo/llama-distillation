import os
import json
import re

PROMPT_ORIGIN_DIR = "./promptOrigin"
TRAIN_FULL_PATH = "./promptTrain/train_data_full.jsonl"
TRAIN_BLOCK_PATH = "./promptTrain/train_data_block.jsonl"

def split_conversation_blocks(text: str):
    pattern = r'(The traveler: .*?\"|The mystical llama: .*?\")'
    parts = re.split(pattern, text, flags=re.DOTALL)
    header = parts[0].strip()
    dialogue = [p.strip() for p in parts[1:] if p.strip()]

    blocks = []
    full_prefix = header + "\n"
    isFirst = True
    for i in range(0, len(dialogue) - 1, 2):
        traveler_line = dialogue[i]
        llama_line = dialogue[i + 1]
        full_prefix += traveler_line + "\n"
        if not isFirst:
            blocks.append({
                "input": full_prefix.strip(),
                "expected": llama_line.replace("The mystical llama: ", "").split("\"")[0].strip() + "\""
            })
        isFirst = False
        full_prefix += llama_line + "\n"
    return blocks

def generate_train():
    os.makedirs("./promptTrain", exist_ok=True)

    with open(TRAIN_FULL_PATH, 'w', encoding='utf-8') as full_file, \
         open(TRAIN_BLOCK_PATH, 'w', encoding='utf-8') as block_file:

        for filename in os.listdir(PROMPT_ORIGIN_DIR):
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(PROMPT_ORIGIN_DIR, filename)
            with open(path, 'r', encoding='utf-8') as f:
                full_text = f.read().strip()

            # 줄바꿈 문자를 유지한 멀티라인 문자열로 저장
            json.dump({"text": full_text}, full_file, ensure_ascii=False)
            full_file.write("\n")

            # 블록 분리 저장
            blocks = split_conversation_blocks(full_text)
            for block in blocks:
                json.dump(block, block_file, ensure_ascii=False)
                block_file.write("\n")

if __name__ == "__main__":
    generate_train()
