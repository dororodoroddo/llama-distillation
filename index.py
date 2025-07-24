import sys
from makePrompt import generate_train
from distillation import train_and_convert
# from evaluate import evaluate_model
from tokenize_and_save import tokenize_and_save

def main():
    if len(sys.argv) < 2:
        print("사용법: python index.py [make|distill|evaluate|all]")
        return

    command = sys.argv[1].lower()

    if command == "make":
        print("[1] makePrompt 실행 중...")
        generate_train()
        print("[1.5] tokenize_and_save 실행 중...")
        tokenize_and_save()
    elif command == "distill":
        print("[2] distillation 실행 중...")
        train_and_convert()
    # elif command == "evaluate":
    #     print("[3] evaluate 실행 중...")
    #     evaluate_model()
    elif command == "all":
        print("[1] makePrompt 실행 중...")
        generate_train()
        print("[1.5] tokenize_and_save 실행 중...")
        tokenize_and_save()
        print("[2] distillation 실행 중...")
        train_and_convert()
        # print("[3] evaluate 실행 중...")
        # evaluate_model()
    else:
        print("알 수 없는 명령입니다. [make|distill|evaluate|all] 중 하나를 입력하세요.")

if __name__ == "__main__":
    main()
