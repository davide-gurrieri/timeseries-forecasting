# Fix randomness and hide warnings
SEED = 42

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

import numpy as np

np.random.seed(SEED)

import logging

import random

random.seed(SEED)

# Import tensorflow
import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)
print(tf.__version__)

# Import other libraries
import statsmodels.graphics.tsaplots.plot_acf as plot_acf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
