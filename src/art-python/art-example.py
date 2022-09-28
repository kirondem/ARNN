#!/usr/bin/env python

import os
import pandas as pd

from train_art import data_train
from test_art import data_test

PATH  =  os.path.dirname(os.path.abspath(__file__))

train_data = pd.read_csv(os.path.join(PATH, 'sample-data/train-example.csv')).values
test_data = pd.read_csv(os.path.join(PATH,'sample-data/test-example.csv')).values
x = train_data[:,1:3]
y = test_data[:,1:3]


r = 0.9
Tmatrix = data_train(x,rho=r) #,beta=0.000001,alpha=1.0,nep=1)

print(Tmatrix)
T = data_test(y,Tmatrix,rho=r) #,beta=0.000001,alpha=1.0,nep=1)
print(T)