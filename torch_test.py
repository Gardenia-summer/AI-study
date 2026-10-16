import torch

print("PyTorch版本：", torch.__version__)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

c = a + b

print("张量a：", a)
print("张量b：", b)
print("计算结果：", c)
print("CUDA可用：", torch.cuda.is_available())