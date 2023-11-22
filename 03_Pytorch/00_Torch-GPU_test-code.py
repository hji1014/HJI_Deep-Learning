import torch
torch.cuda.is_available()

torch.cuda.device_count()

torch.cuda.get_device_name(0)
# torch.cuda.get_device_name(1)

# GPU가 사용 가능하면 cuda, 아니면 cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# .to() 메소드를 사용하여 텐서나 모델 등을 장치로 이동
# model = NeuralNetwork().to(device)
