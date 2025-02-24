import torch
import torchvision.models as models

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)

model = model.to("xpu")
data = data.to("xpu")

with torch.no_grad():
    model(data)

print("Execution finished")