from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 체크포인트 경로
checkpoint_path = "my-check/checkpoint-100"

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 테스트 프롬프트
fixed_text = ''
# fixed_text += """[{"no":0,"card":"The Fool","upright":["Adventure","Innocence"],"reversed":["Recklessness","Foolishness"]},{"no":1,"card":"The Magician","upright":["Creativity","Ingenuity"],"reversed":["Timidity","Deception"]},{"no":2,"card":"The High Priestess","upright":["Knowledge","Wisdom"],"reversed":["Cruelty","Rudeness"]},{"no":3,"card":"The Empress","upright":["Abundance","Motherhood"],"reversed":["Excess","Vanity"]},{"no":4,"card":"The Emperor","upright":["Responsibility","Fatherhood"],"reversed":["Arrogance","Domination"]},{"no":5,"card":"The Hierophant","upright":["Teaching","Generosity"],"reversed":["Pettiness","Laziness"]},{"no":6,"card":"The Lovers","upright":["Romance","Pleasure"],"reversed":["Jealousy","Betrayal","Heartbreak"]},{"no":7,"card":"The Chariot","upright":["Progress","Victory"],"reversed":["Rampage","Frustration","Defeat"]},{"no":8,"card":"Strength","upright":["Power","Courage"],"reversed":["Instinct","Arrogance"]},{"no":9,"card":"The Hermit","upright":["Exploration","Thoughtfulness"],"reversed":["Gloominess","Isolation","Greed"]},{"no":10,"card":"Wheel of Fortune","upright":["Opportunity","Temporary Luck"],"reversed":["Misjudgment","Misfortune"]},{"no":11,"card":"Justice","upright":["Balance","Fairness"],"reversed":["Imbalance","Prejudice","Injustice"]},{"no":12,"card":"The Hanged Man","upright":["Self-Sacrifice","Patience"],"reversed":["Futile Sacrifice","Blindness"]},{"no":13,"card":"Death","upright":["Transformation","Farewell"],"reversed":["Resistance to Change","Stagnation"]},{"no":14,"card":"Temperance","upright":["Harmony","Steadiness"],"reversed":["Wastefulness","Instability"]},{"no":15,"card":"The Devil","upright":["Selfishness","Bondage","Corruption"],"reversed":["Awakening from a Vicious Cycle"]},{"no":16,"card":"The Tower","upright":["Destruction","Ruin"],"reversed":["Necessary Collapse"]},{"no":17,"card":"The Star","upright":["Hope","Aspiration"],"reversed":["Disillusionment","Sorrow"]},{"no":18,"card":"The Moon","upright":["Anxiety","Ambiguity","Chaos"],"reversed":["Anxiety Relief","Clarity","End of Confusion"]},{"no":19,"card":"The Sun","upright":["Bright Future","Contentment"],"reversed":["Delay","Failure"]},{"no":20,"card":"Judgement","upright":["Revival","Improvement"],"reversed":["Irrecoverable Fall","Regret"]},{"no":21,"card":"The World","upright":["Completion","Perfection"],"reversed":["Incompletion","Ambiguity"]}]
# """
fixed_text += """Today is 2025. 7. 26.

"""
prompt = fixed_text + '''The mystical llama: "What is your concern?"
The Traveler: "How is my job luck this year? The drawn card is 'I - The Magician (reversed)'."
The mystical llama: "'''

# 토크나이즈
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 응답 생성
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

# 출력 디코딩
print("\n=== Generated Response ===")
print(tokenizer.decode(output[0], skip_special_tokens=True))
