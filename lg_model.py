# Tutorial 1. Logistic Model.

import gzip
import torch
import tenseal as ts
import pandas as pd
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt

def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    delim = int(len(x)*test_ratio)
    test_idx, train_idx = idxs[:delim], idxs[delim:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

def get_txt(file):
    ff = open(file, "r")
    dd = ff.read()
    d_l = dd.replace("\n", "\t").split("\t")
    return d_l

def train_data():
    print("Process Starts!")
    N_GENE = 20390
    N_SAMPLE = 3000
    input_file = gzip.open('CHALLENGE_DATA/genotypes.txt.gz','r')
    print("Geno file read!")
    r_file = input_file.read()
    data_l = r_file.decode("utf-8")
    data_l=data_l.replace("\n", "\t").split("\t")
    print("Geno file split!")
    data_l = np.delete(np.array(data_ll), -1)
    print("Geno file to array!")
    data_l = np.delete(data_l.reshape((N_GENE, N_SAMPLE+4)), range(4), axis=1)
    print("Geno file reshaped!")
    data_l = torch.tensor(data_l.T).float()
    print(f"Genotype data processed: {data_l.shape}")

    phe = []
    for i in range(5):
        phe.append(get_txt('CHALLENGE_DATA/phenotypes_'+str(i+1)+'.txt'))

    index = get_txt('CHALLENGE_DATA/genotype_sample_ids.list')
    train_y = np.zeros((len(phe),len(index)-1))
    for i in range(len(phe)):
        for j in range(len(index)-1):
            p = phe[i].index(index[j])
            # print(p)
            train_y[i,j] = phe[i][p+1]
    for yy in train_y:
        yy = (yy-yy.mean())/yy.std()
    train_y = torch.tensor(train_y.T).float()
    print(f"Phenotype data processed: {train_x.shape}")

    return split_train_test(data_l, train_y, test_ratio=0.3)

x_train, y_train, x_test, y_test = train_data()

print("############# Data summary #############")
print(f"x_train has shape: {x_train.shape}")
print(f"y_train has shape: {y_train.shape}")
print(f"x_test has shape: {x_test.shape}")
print(f"y_test has shape: {y_test.shape}")
print("#######################################")

