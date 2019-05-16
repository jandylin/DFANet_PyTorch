import torch
import torchvision.datasets
from model.backbone import backbone


x = torch.rand([1, 3, 224, 224]).cuda()
print(x.shape)
model = backbone().cuda()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
x, enc2, enc3, enc4, fc, fca = model(x)
print(x.shape)
print(enc2.shape)
print(enc3.shape)
print(enc4.shape)
print(fc.shape)
print(fca.shape)
