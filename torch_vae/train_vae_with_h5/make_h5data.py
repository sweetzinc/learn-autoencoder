#%%
import os
import pandas as pd
import json 
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
import h5py
from torch_vae.lightningdata_engchar import HandwrittenCharDataset


if __name__ == "__main__" : 
    if os.name == 'nt':
        data_dir = r'C:\docker_share\downloaded\EngLetters'
        output_dir = r'C:\docker_share\asn_workinprogress'
    else:
        data_dir = '/mounted_data/downloaded/EngLetters'
        output_dir = '/mounted_data/asn_workinprogress'

    dataset = HandwrittenCharDataset(dataset_dir=data_dir, csv_file='labels.csv', transform=None)

    mapping = dataset.mapping
    df = dataset.data 
    indices = df[df['label']=='A'].index.values 
    imgs = []
    for idx in indices : 
        img, label = dataset[idx]
        imgs.append(img)

    imgs_array = np.array(imgs, dtype=np.float32)

    output_file = os.path.join(output_dir, 'images_A.h5')
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('images', data=imgs_array)
        print(f"Saved images to {output_file}")
# %%
