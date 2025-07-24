import os

print("여러번 저장하세요. 끝낼 때는 'zzz'이라고 입력하세요:\n")
allEnd = False
while True: 
# 저장할 디렉토리 경로
    directory = './promptOrigin/'
    os.makedirs(directory, exist_ok=True)

    # 기존 파일 번호 구하기
    existing_files = [f for f in os.listdir(directory) if f.endswith('.txt') and f[:-4].isdigit()]
    existing_numbers = sorted([int(f[:-4]) for f in existing_files])
    next_number = (existing_numbers[-1] + 1) if existing_numbers else 0
    filename = f"{next_number}.txt"
    filepath = os.path.join(directory, filename)

    # 고정 텍스트
    fixed_text = """No. | Card              | Upright Meaning                       | Reversed Meaning
----|-------------------|----------------------------------------|-----------------------------------------------
00  | The Fool          | Adventure, Innocence                  | Recklessness, Foolishness
01  | The Magician      | Creativity, Ingenuity                 | Timidity, Deception
02  | The High Priestess| Knowledge, Wisdom                     | Cruelty, Rudeness
03  | The Empress       | Abundance, Motherhood                 | Excess, Vanity
04  | The Emperor       | Responsibility, Fatherhood            | Arrogance, Domination
05  | The Hierophant    | Teaching, Generosity                  | Pettiness, Laziness
06  | The Lovers        | Romance, Pleasure                     | Jealousy, Betrayal, Heartbreak
07  | The Chariot       | Progress, Victory                     | Rampage, Frustration, Defeat
08  | Strength          | Power, Courage                        | Instinct, Arrogance
09  | The Hermit        | Exploration, Thoughtfulness           | Gloominess, Isolation, Greed
10  | Wheel of Fortune  | Opportunity, Temporary Luck           | Misjudgment, Misfortune
11  | Justice           | Balance, Fairness                     | Imbalance, Prejudice, Injustice
12  | The Hanged Man    | Self-Sacrifice, Patience              | Futile Sacrifice, Blindness
13  | Death             | Transformation, Farewell              | Resistance to Change, Stagnation
14  | Temperance        | Harmony, Steadiness                   | Wastefulness, Instability
15  | The Devil         | Selfishness, Bondage, Corruption      | Awakening from a Vicious Cycle
16  | The Tower         | Destruction, Ruin                     | Necessary Collapse
17  | The Star          | Hope, Aspiration                      | Disillusionment, Sorrow
18  | The Moon          | Anxiety, Ambiguity, Chaos             | Anxiety Relief, Clarity, End of Confusion
19  | The Sun           | Bright Future, Contentment            | Delay, Failure
20  | Judgement         | Revival, Improvement                  | Irrecoverable Fall, Regret
21  | The World         | Completion, Perfection                | Incompletion, Ambiguity

You are a mystical guide of the tarot, known as the llama.
You interpret vague concerns through the tarot and help the traveler refine their question — guiding them toward the answer that already lies within.

Today is 2025. 7. 26.

"""

    # 사용자 입력 (종료어: vvv)
    print("아래에 여러 파일을을 입력하세요. 끝낼 때는 'vvv'이라고 입력하세요:\n")
    user_lines = []
    while True:
        line = input()
        if line.strip() == 'vvv':
            break
        
        if line.strip() == 'zzz':
            allEnd = True
            break
        user_lines.append(line)

    if allEnd:
        break
    user_text = '\n'.join(user_lines)

    # 파일로 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_text + user_text + '\n')

    print(f"\n✅ 파일이 저장되었습니다: {filepath}")
