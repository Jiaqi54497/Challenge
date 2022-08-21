# Don't need anymore. Done in jupter lab.

import gzip
import numpy as np
from tqdm import tqdm

input_file = gzip.open('CHALLENGE_DATA/genotypes.txt.gz','r')
r_file = input_file.read()
data_l = r_file.decode("utf-8")
data_ll=data_l.split("\n")

t_data = np.zeros(0)
for l in tqdm(data_ll):
    t_data = np.concatenate([t_data, np.array(l.split("\t")[4:])])
np.save(t_data, train_x)

