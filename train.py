from model.dfanet import DFANet
import torch

x = torch.rand([2, 3, 1024, 1024], dtype=torch.float32)
x = x.cuda()
print(x.shape)

model = DFANet(pretrained_backbone=False)
model = model.cuda()
out = model(x)
print(out.shape)
