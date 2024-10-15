"""
Reference:
https://huggingface.co/docs/transformers/en/model_doc/convnextv2 # Documentation
https://huggingface.co/facebook/convnextv2-tiny-22k-224 # Pretrained model
"""
#%%
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
import h5py
from torch.utils.data import Dataset, DataLoader
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and preprocessor
# model_name = "facebook/convnextv2-tiny-22k-224"
model_name="microsoft/swinv2-tiny-patch4-window8-256"
cache_dir = "/mounted_data/downloaded/pretrained" 
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
preprocessor = AutoImageProcessor.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# # Get the number of parameters in the model
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")

#%% # Define the transformations 
size = (
    preprocessor.size["shortest_edge"]
    if "shortest_edge" in preprocessor.size
    else (preprocessor.size["height"], preprocessor.size["width"])
)
# Set up, adjust normalize values if needed
tvtransforms = Compose([RandomResizedCrop(size), 
                        ToTensor(), 
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) 

#%% # To extract features from an image
def extract_features(image_path, output_type='last_hidden_state', transformations=None):

    inputs = {}
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    if transformations:
        inputs['pixel_values'] = transformations(image).unsqueeze(0).to(device)
    else:
        inputs = preprocessor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    if output_type == 'last_hidden_state':
        features = outputs.last_hidden_state.squeeze().cpu().numpy()
    elif output_type == 'pooler_output':
        if hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output.squeeze().cpu().numpy()
        else:
            # If pooler_output is not available, use mean pooling of last_hidden_state
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    elif output_type == 'all_hidden_states':
        features = [hs.squeeze().cpu().numpy() for hs in outputs.hidden_states]
    else:
        raise ValueError(f"Unknown output_type: {output_type}")
    
    return features

#%%
image_path = "/mounted_data/balloon/train/120853323_d4788431b9_b.jpg"
features = extract_features(image_path, output_type='pooler_output', 
                            transformations=tvtransforms)
print(f"Extracted features shape: {features.shape}")

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path

# Define the dataset and dataloader
image_dir = "/mounted_data/balloon/train"
dataset = ImageDataset(image_dir, transform=tvtransforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# File to save the features
h5_file_path = "/mounted_data/features.h5"

# Extract features and save to h5 file
with h5py.File(h5_file_path, 'a') as h5_file:
    for images, image_paths in dataloader:
        for image, image_path in zip(images, image_paths):
            features = extract_features(image_path, output_type='pooler_output', transformations=None)
            image_name = os.path.basename(image_path)
            h5_file.create_dataset(image_name, data=features)
            h5_file['features'].resize((h5_file['features'].shape[0] + 1), axis=0)
            h5_file['features'][-1] = features