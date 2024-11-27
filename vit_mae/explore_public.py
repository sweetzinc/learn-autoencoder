#%%
import torch
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEModel
from PIL import Image
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
import os
from matplotlib import pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and preprocessor
model_name="facebook/vit-mae-base"
cache_dir = "/mounted_data/downloaded/pretrained" 
model = ViTMAEForPreTraining.from_pretrained(model_name, cache_dir=cache_dir) 
bare_model = ViTMAEModel.from_pretrained(model_name, cache_dir=cache_dir)
preprocessor = AutoImageProcessor.from_pretrained(model_name)
# %%

imgfpath = '/mounted_data/downloaded/ai.png'
img = Image.open(imgfpath).convert("RGB")
fig, ax = plt.subplots(1, 1, figsize=(2,2)); ax.imshow(img); ax.axis('off')

bare_model_inputs = preprocessor(images=img, return_tensors="pt") 
bare_model_outputs = bare_model(**bare_model_inputs)
# %%
