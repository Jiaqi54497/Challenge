import gzip

with gzip.open('CHALLENGE_DATA/genotypes.txt.gz','rt') as f:
    for line in f:
        print('got line', line)
