print("suup") 
import pandas as pd
import numpy as np
import random as rnd
# try doing this? import matplotlib as mpl
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
print(train_df.columns.values)

# preview the data
#nice job kent
#here's why it needed to be moved: ../input/'train.csv' takes us up a level, 
# so we would've need to go up a level and have a folder named input (per 
#the source code) where train.csv


train_df.head()

print("working?")

print("Did the change come through?")
