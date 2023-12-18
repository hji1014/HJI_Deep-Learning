"""
code : skoltech_image_cap/On-the-Automatic-Generation-of-Medical-Imaging-Reports/ModelTrainingAndEvaluating.ipynb
"""

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"     # 선택한 GPU로만 학습
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import argparse
from models import EncoderCNN, SentenceLSTM, WordLSTM
from collections import Counter

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


from dataloader import get_loader
from score import evalscores
from torchvision import transforms
from torch import nn
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

# ================================================================================================================ #

from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
import json
import pandas as pd

# ================================================================================================================ #

from collections import Counter

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

# ================================================================================================================ #

# !nvidia-smi

from train import script

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type = int, default = 224, help = 'size to which image is to be resized')
parser.add_argument('--crop_size', type = int, default = 224, help = 'size to which the image is to be cropped')
parser.add_argument('--device_number', type = str, default = "0", help = 'which GPU to run experiment on')

parser.add_argument('--int_stop_dim', type = int, default = 64, help = 'intermediate state dimension of stop vector network')
parser.add_argument('--sent_hidden_dim', type = int, default = 512, help = 'hidden state dimension of sentence LSTM')
parser.add_argument('--sent_input_dim', type = int, default = 1024, help = 'dimension of input to sentence LSTM')
parser.add_argument('--word_hidden_dim', type = int, default = 512, help = 'hidden state dimension of word LSTM')
parser.add_argument('--word_input_dim', type = int, default = 512, help = 'dimension of input to word LSTM')
parser.add_argument('--att_dim', type = int, default = 64, help = 'dimension of intermediate state in co-attention network')
parser.add_argument('--num_layers', type = int, default = 1, help = 'number of layers in word LSTM')

parser.add_argument('--lambda_sent', type = int, default = 1, help = 'weight for cross-entropy loss of stop vectors from sentence LSTM')
parser.add_argument('--lambda_word', type = int, default = 1, help = 'weight for cross-entropy loss of words predicted from word LSTM with target words')

parser.add_argument('--batch_size', type = int, default = 8, help = 'size of the batch')
parser.add_argument('--shuffle', type = bool, default = True, help = 'shuffle the instances in dataset')
parser.add_argument('--num_workers', type = int, default = 0, help = 'number of workers for the dataloader')
parser.add_argument('--num_epochs', type = int, default = 1, help = 'number of epochs to train the model')
parser.add_argument('--learning_rate_cnn', type = int, default = 1e-5, help = 'learning rate for CNN Encoder')
parser.add_argument('--learning_rate_lstm', type = int, default = 5e-3, help = 'learning rate for LSTM Decoder')

parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

args = parser.parse_args('')

# ================================================================================================================ #

# os.environ["CUDA_VISIBLE_DEVICES"]= args.device_number
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(args.device)
print(args)

# ==================================================== 학습 시작 =================================================== #

# pretrained weights 파일이 없어서 불러오지 못함 (models.py에서 확인)
args, val_loader, encoderCNN, sentLSTM, wordLSTM, vocab, hypotheses, references = script(args)
