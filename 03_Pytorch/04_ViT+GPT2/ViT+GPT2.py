"""
Image Captioning : ViT + GPT2
datasets : Flickr8k
ref : https://www.kaggle.com/code/burhanuddinlatsaheb/image-captioning-vit-gpt2/notebook
"""


"""
[1. Imports]
"""
import os

import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from transformers import AutoTokenizer, GPT2Config, default_data_collator


if torch.cuda.is_available():

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


"""
[2. Hyperparameters]
"""
os.environ["WANDB_DISABLED"] = "true"
class config :
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    NUM_WORKERS = mp.cpu_count()
    EPOCHS = 2
    IMG_SIZE = (224, 224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95


"""
[3. Helper Functions]
- There are Two helper functions:
1) The first function is to build special tokens while tokenizing the captions
2) The second function is used to compute the ROUGE-2 metrics as we are working with Transformers
"""
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


"""
[4. Dataset]
- Feature Extractor and Tokenizer
    1) The Feature extractor is loaded using ViTFeatureExtractor
    2) The tokenizer for GPT2 is loaded using the AutoTokenizer
- Transforms and Dataframe
    1) Resizing the image to (224,224)
    2) Normalizing the image
    3) Converting the image to Tensor
- Dataset Class
    1) We read the image using the Image function of PIL library
    2) The image is transformed using the transformed defined above
    3) The transformed image is passed through the feature extractor to extract the pixel values from the image
    4) The captions are loaded from the dataframe
    5) The captions are tokenized
    6) The tokenized captions are padded to max length
    7) The images and tokenized captions are returned
- Train and Validation dataset
"""
# - Feature Extractor and Tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token

# - Transforms and Dataframe
transforms = transforms.Compose(
    [transforms.Resize(config.IMG_SIZE),
     transforms.ToTensor(),
     transforms.Normalize(mean=0, std=1),
     #transforms.Normalize(mean=config.MEAN, std=config.STD)
     #transforms.Normalize(mean=0.5, std=0.5)
     ]
)

df = pd.read_csv("C:/Users/user02/PycharmProjects/py_37/01_DL_practice/01_Tensorflow_2.x/03_Basic_NIC/show_and_tell_implementation-master/Flickr8k_text/captions.txt")
train_df, val_df = train_test_split(df, test_size=0.2)      # 여기서 막 섞으면 안될 것 같은데?
df.head()

# - Dataset Class
class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 50

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        captions = self.tokenizer(caption,
                                  padding='max_length',
                                  max_length=self.max_length).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(captions)}
        return encoding

# - Train and Validation dataset
train_dataset = ImgDataset(train_df,
                           root_dir="C:/Users/user02/PycharmProjects/py_37/01_DL_practice/01_Tensorflow_2.x/03_Basic_NIC/show_and_tell_implementation-master/Flickr8k_Dataset",
                           tokenizer=tokenizer,
                           feature_extractor=feature_extractor,
                           transform=transforms)
val_dataset = ImgDataset(val_df,
                         root_dir="C:/Users/user02/PycharmProjects/py_37/01_DL_practice/01_Tensorflow_2.x/03_Basic_NIC/show_and_tell_implementation-master/Flickr8k_Dataset",
                         tokenizer=tokenizer,
                         feature_extractor=feature_extractor,
                         transform=transforms)


"""
[5. Model Building]
- Model Initialization
"""
# - Model Initialization
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size       # make sure vocab size is set correctly -> 디코더의 voca 사이즈를 쓰는게 맞나?
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4                                      # set beam search parameters


"""
[6. Training]
- Training Arguments
- Training using Seq2SeqTrainer
"""
# - Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='VIT_large_gpt2',
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,
    save_steps=2048,
    warmup_steps=1024,
    learning_rate=5e-5,
    #max_steps=1500, # delete for full training
    num_train_epochs=config.EPOCHS, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1,
)

# - Training using Seq2SeqTrainer
trainer = Seq2SeqTrainer(                   # instantiate trainer
    tokenizer=feature_extractor,
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)
trainer.train()                             # 학습 시작

trainer.save_model('VIT_large_gpt2')        # fine-tuned 모델 저장


"""
[7. Predictions]
"""
# 하나 예시
img = Image.open("C:/Users/user02/PycharmProjects/py_37/01_DL_practice/01_Tensorflow_2.x/03_Basic_NIC/show_and_tell_implementation-master/Flickr8k_Dataset/1001773457_577c3a7d70.jpg").convert("RGB")
#plt.imshow(img)
#plt.show()

generated_caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cuda"))[0])
print('\033[96m' +generated_caption[:85]+ '\033[0m')
#print(generated_caption[:85])
