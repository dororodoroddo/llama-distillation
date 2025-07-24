import os

# 처리 대상 디렉토리
PROMPT_DIR = "./promptOrigin"

# 제거할 헤더 패턴 일부 (시작 문장 포함 여부 확인용)
HEADER_KEYWORD = "You are a mystical guide of the tarot"
HEADER_END_MARK = "Today is"

def strip_prompt_header(text: str) -> str:
    """
    의미표와 설정 텍스트가 끝나는 지점 이후 텍스트만 반환
    """
    if HEADER_KEYWORD not in text:
        return text  # 헤더가 없다면 그대로 반환

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith(HEADER_END_MARK):
            # 이후 줄부터 반환
            return "\n".join(lines[idx + 1:]).lstrip()
    return text

def process_all_txt_files():
    for filename in os.listdir(PROMPT_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(PROMPT_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                original = f.read()

            stripped = strip_prompt_header(original)

            # 덮어쓰기
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(stripped)

            print(f"[✓] Header removed: {filename}")

if __name__ == "__main__":
    process_all_txt_files()
