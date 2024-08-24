import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.nn as nn
import gradio as gr

# Use GPU if available
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

#load the saved model
MODEL_PATH = "model/lettuce_npk.pth"
model = torch.load(MODEL_PATH).to(device)
model.eval()

def prediction(img):
    t = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize(224, antialias=True)
    ])
    new_img = t(img)
    model.eval()
    with torch.no_grad():
        pred = model(torch.stack([new_img]).to(device)).cpu().detach().numpy()[0]
    class_label = np.argmax(pred)
    return class_names[np.argmax(pred)], pred[class_label]

classes = ["-K", "-N", "-P", "FN"]
class_descriptions = {
    "-K": "Potassium deficiency", 
    "-N": "Azote deficiency",
    "-P": "Phosphorus deficiency",
    "FN": "Healthy"
}
class_names = dict(zip(
    range(len(classes)), 
    [class_descriptions[c] for c in sorted(classes)]
))

app = gr.Interface(prediction, gr.Image(), "text")
app.launch(share=True)