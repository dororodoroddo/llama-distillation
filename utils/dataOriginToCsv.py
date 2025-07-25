import csv
import os

# 저장할 CSV 파일 경로
output_csv = './csv/merged_output.csv'

# CSV 헤더
fieldnames = ['number', 'topic', 'used card', 'content']

# 파일이 있는 디렉토리 (현재 디렉토리라면 '.')
input_dir = './promptOrigin/'

with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(1, 101):
        filename = os.path.join(input_dir, f"{i}.txt")
        if not os.path.exists(filename):
            print(f"파일 없음: {filename}")
            continue

        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        writer.writerow({
            'number': i,
            'topic': '',
            'used card': '',
            'content': content
        })

print("CSV 파일 생성 완료:", output_csv)
