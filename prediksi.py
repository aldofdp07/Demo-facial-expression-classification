import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
#from modelDens import model

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 7)


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

PATH = "D:\Smt7\Skripsi\demo\modd.pth"

device = torch.device("cpu")
mod = model
mod.load_state_dict(torch.load(PATH, map_location=device))

# Make sure to call input = input.to(device) on any input tensors that you feed to the model

# mod= torch.load(PATH)
# mod.eval()
transfo =transforms.Compose([transforms.Resize(254),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.RandomHorizontalFlip(),
                             transforms.Normalize(mean, std)])

def predict(image_path):
    was_training = mod.training
    mod.eval()
    images_so_far = 0
    
    with torch.no_grad():
        img = Image.open(image_path)
        
        batch_t = torch.unsqueeze(transfo(img),0)
        batch_t = batch_t.to(device)
        
        outputs = mod(batch_t)
        print(outputs[0])
        _,preds = torch.max(outputs,1)
        
        mod.train(mode=was_training)
        
    return(preds)
        
        