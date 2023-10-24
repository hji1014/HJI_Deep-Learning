import pandas as pd
import urllib3
import zipfile
import shutil
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

print(tf.__version__)

