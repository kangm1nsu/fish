# 1. 라이브러리 설치
# pip install numpy - 넘파이 설치
# pip install pandas - 판다스 설치
# pip3 install mariadb SQLAlchemy - SQLAlchemy 설치

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sqlalchemy as db
import matplotlib.pyplot as plt


bl = pd.read_csv('C:/workspace/download/fish/bream_length.csv', header=None);
bream_length = bl.to_numpy().flatten()

bw = pd.read_csv('C:/workspace/download/fish/bream_weight.csv', header=None);
bream_weight = bw.to_numpy().flatten()

sl = pd.read_csv('C:/workspace/download/fish/smelt_length.csv', header=None);
smelt_length = sl.to_numpy().flatten()

sw = pd.read_csv('C:/workspace/download/fish/smelt_weight.csv', header=None);
smelt_weight = sw.to_numpy().flatten()
#================================
bream_data = np.column_stack((bream_length, bream_weight));

smelt_data = np.column_stack((smelt_length, smelt_weight));
#================================
plt.scatter(bream_data[:,0], bream_data[:,1])
plt.scatter(smelt_data[:,0], smelt_data[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

#================================
fish_data = np.concatenate((bream_data, smelt_data))
print(fish_data.shape) # [[길이, 무게]]

#================================
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
#================================
fish_target = fish_target.reshape((-1,1)) # (49,1)
fishes = np.hstack((fish_target, fish_data))
print(fishes)

#================================
index = np.arange(49) # 도미 : 0~34, 빙어 : 35~48
np.random.shuffle(index)
print(index)

#================================
# 훈련 데이터 - train data,
train_input = fish_data[index[:35]]
train_target = fish_target[index[:35]]

# 테스트 데이터 - test data
test_input = fish_data[index[35:]]
test_target = fish_target[index[35:]]

#================================
plt.scatter(train_input[:,0], train_input[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

#Db
engine = db.create_engine("mariadb+mariadbconnector://python:python1234@127.0.0.1:3306/pythondb")

#================================
train_data = pd.DataFrame(fishes[index[:35]], columns=["train_target","train_lentgh", "train_weight"])
train_data.to_sql("train", engine, index=False, if_exists="replace" )

#================================
test_data = pd.DataFrame(fishes[index[35:]], columns=["train_target","train_lentgh", "train_weight"])
test_data.to_sql("test", engine, index=False, if_exists="replace" )