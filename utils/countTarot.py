import os
import re
from collections import defaultdict

# 카드별 방향 횟수를 저장할 딕셔너리
# 예: stats['XIII - Death']['upright'] += 1
def count():
    stats = defaultdict(lambda: {'upright': 0, 'reversed': 0})

    # 파일이 저장된 디렉토리 경로 (현재 경로라면 ".")
    base_dir = "./promptOrigin/"

    # 대상 파일 이름들 (1.txt ~ 100.txt)
    for i in range(1, 101):
        file_path = os.path.join(base_dir, f"{i}.txt")
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            # 카드 추출 정규표현식: 'XIII - Death (upright)' 또는 'VI - The Lovers (reversed)' 형태
            matches = re.findall(r"'([0IVXLCDM]+ - [^']+?) \((upright|reversed)\)'", content)
            if len(matches) < 4:
                print(file_path + str(matches))

            for card_name, direction in matches:
                stats[card_name][direction] += 1

    # 결과 출력
    print(f"{'Card':40} | Upright | Reversed")
    print("-" * 60)
    for card in sorted(stats.keys()):
        upright = stats[card]['upright']
        reversed_ = stats[card]['reversed']
        print(f"{card:40} | {upright:^7} | {reversed_:^8}")
    total_upright = sum(c['upright'] for c in stats.values())
    total_reversed = sum(c['reversed'] for c in stats.values())
    print("-" * 70)
    print(f"{'Total':40} | {total_upright:^7} | {total_reversed:^8} | {total_upright + total_reversed:^5}")


if __name__ == "__main__":
    count()
