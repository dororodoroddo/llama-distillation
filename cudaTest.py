import torch
print(torch.cuda.is_available())  # True가 나오면 성공
print(torch.cuda.get_device_name(0))  # 'NVIDIA GeForce RTX 2060 SUPER' 출력
print(torch.version.cuda)
