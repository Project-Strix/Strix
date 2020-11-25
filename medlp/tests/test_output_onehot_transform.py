import torch  
from medlp.models.cnn.utils import output_onehot_transform

output = {}
output["pred"] = torch.rand([5,2])
output["label"] = torch.rand([5])

pred, label = output_onehot_transform(output, n_classes=2)
print(output['pred'].shape, output['label'].shape)
print(pred.shape, label.shape)