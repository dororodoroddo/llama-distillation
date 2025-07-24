# evaluate.py
# promptTest 기반으로 학습된 모델의 예측 결과를 평가합니다.

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_DIR = config["output_dir"]
TEST_FILE = config["test_file"]
MAX_LEN = config["max_length"]


def evaluate_model():
    print("[evaluate] 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True).cuda()
    model.eval()

    print("[evaluate] 테스트셋 로딩 중...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    correct = 0
    total = 0

    print("[evaluate] 예측 시작...")
    for example in tqdm(dataset):
        input_text = example["input"]
        expected = example["expected"].strip()

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 모델 출력 중 마지막 llama 응답만 추출
        if "The mystical llama:" in decoded:
            result = decoded.split("The mystical llama:")[-1].strip()
        else:
            result = decoded.strip()

        print("\n[INPUT]\n", input_text)
        print("[EXPECTED]\n", expected)
        print("[RESULT]\n", result)

        # 간단한 포함 기반 평가
        if expected[:10] in result:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"\n[evaluate] 정확도: {acc*100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate_model()
