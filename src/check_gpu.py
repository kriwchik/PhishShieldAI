import torch

print("CUDA доступна:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Устройство:", torch.cuda.get_device_name(0))
    print("Количество GPU:", torch.cuda.device_count())
else:
    print("Используется CPU")
