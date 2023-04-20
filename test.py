# 示例代码
import torch

# model

model = torch.hub.load("./","custom","apex/weight/best.pt", source="local")

# Images
img = "./apex/data/train/images/_5__640x640__jpg.rf.0f8063a8cdea2bba63d8c2d9126d2277.jpg"

# Inference
results = model(img)

# results
results.show()