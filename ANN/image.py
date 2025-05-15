import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

path = "files/letters.csv"
df = pd.read_csv(path)
train = df[:60000]
test = df[60000:70000]
train.to_csv("files/letter_train.csv", index = False)
test.to_csv("files/letter_test.csv", index = False)
