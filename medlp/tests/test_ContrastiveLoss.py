import torch
from medlp.models.cnn.losses import ContrastiveLoss


loss = ContrastiveLoss()
output1 = torch.randn(5, 128)
output2 = torch.randn(5, 128)
target1 = torch.Tensor([1, 1, 1, 0, 0])
target2 = torch.Tensor([1, 1, 0, 1, 0])

loss(output1, output2, target1, target2)
