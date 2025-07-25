import subprocess

model_dir='./my-check/checkpoint-100/'
output_dir='./my-check/outputGguf/'

def convert_to_gguf(model_dir, output_dir):
    try:
        print("[→] GGUF 변환 시작...")
        subprocess.run([
            "python", "../android-llama/llama.cpp/convert_hf_to_gguf.py",
             model_dir,
            "--outfile", output_dir + "tiny8_0.gguf",
            "--outtype", "q8_0", #극한의 테스트 - tq2_0, 실사용 - q8_0
            "--model-name tinyllama"
        ], check=True)
        print("[✓] GGUF 변환 완료")
    except subprocess.CalledProcessError as e:
        print(f"[!] GGUF 변환 실패: {e}")


convert_to_gguf(model_dir, output_dir)

# python ../android-llama/llama.cpp/gguf-py/gguf/gguf.py dump ./my-check/outputGguf/tiny8_0.gguf