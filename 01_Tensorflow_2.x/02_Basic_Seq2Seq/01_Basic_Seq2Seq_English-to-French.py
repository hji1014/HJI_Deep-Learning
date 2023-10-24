"""
ref : https://wikidocs.net/24996, https://github.com/ukairia777/tensorflow-nlp-tutorial/blob/main/14.%20Seq2Seq%20(NMT)/14-1.%20char_level_seq2seq.ipynb
"""

import pandas as pd
import urllib3
import zipfile
import shutil
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

print(tf.__version__)

""" Data Downloading """
# http = urllib3.PoolManager()
# url = 'http://www.manythings.org/anki/fra-eng.zip'
# filename = 'fra-eng.zip'
# path = os.getcwd()
# zipfilename = os.path.join(path, filename)
# with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:
#     # Zip File Downloading
#     shutil.copyfileobj(r, out_file)
#
# with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
#     zip_ref.extractall(path)

""" 데이터 확인 및 불필요 feature 제거 """
# 파일 실제/절대 경로 확인 : os.path.realpath(__file__), os.path.abspath(__file__)
lines = pd.read_csv('./01_DL_practice/01_Tensorflow_2.x/02_Basic_Seq2Seq/fra-eng/fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
print('전체 샘플의 개수 :', len(lines))

