import cv2
import mediapipe as mp
import time
import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize, fmin_tnc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from preprocess import preprocess
from handDetectorModule import handDetector
from move import move
from model import LRmodel

data = pd.read_csv("data.csv", index_col=0) #We can create and data with sampleDataPreprocess() function below