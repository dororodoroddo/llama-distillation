# index.py 실행 예시 (CLI)
# python index.py make            # makePrompt만 실행
# python index.py distill         # distillation만 실행
# python index.py evaluate        # evaluate만 실행
# python index.py all             # 전체 파이프라인 실행

import sys
import os
from makePrompt import generate_train
from distillation import train_and_convert
# from evaluate import evaluate_model

def main():
    if len(sys.argv) < 2:
        print("사용법: python index.py [make|distill|all]")
        return

    command = sys.argv[1].lower()

    if command == "make":
        print("[1] makePrompt 실행 중...")
        generate_train()
    elif command == "distill":
        print("[2] distillation 실행 중...")
        train_and_convert()
    # elif command == "evaluate":
    #     print("[3] evaluate 실행 중...")
    #     evaluate_model()
    elif command == "all":
        print("[1] makePrompt 실행 중...")
        generate_train()
        print("[2] distillation 실행 중...")
        train_and_convert()
        # print("[3] evaluate 실행 중...")
        # evaluate_model()
    else:
        print("알 수 없는 명령입니다. [make|distill|evaluate|all] 중 하나를 입력하세요.")

if __name__ == "__main__":
    main()
