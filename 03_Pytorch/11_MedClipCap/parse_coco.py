import warnings
warnings.filterwarnings('ignore')

import torch
import skimage.io as io
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    out_path = f"./data/coco/oscar_split_{clip_model_type}_train.pkl"
    if clip_model_type == "MedCLIPVisionModelViT":
        clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    else:
        clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
    preprocess = MedCLIPProcessor()     # image transformations (convert rgb + resizing + center cropping + normalization)
    clip_model.from_pretrained()
    clip_model.cuda()
    data_ = pd.read_csv("../df_finetune.csv")
    data, val_data = train_test_split(data_, test_size=0.1, shuffle=False)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data.iloc[i]
        img_id = d["image"]
        filename = img_id
        image = Image.open(filename)
        inputs = preprocess(
            text=["opacity left",],
            images=image,
            return_tensors="pt",
            padding=True
            )
        with torch.no_grad():
            outputs = clip_model(**inputs)
            prefix = outputs['img_embeds'].cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 1000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="MedCLIPVisionModelViT", choices=('MedCLIPVisionModel', 'MedCLIPVisionModelViT'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
