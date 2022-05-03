import numpy as np
import matplotlib.pyplot as plt
import time
import os
import D01 as De
import Methods as met
from multiprocessing import Pool

def process_(save_):
    for i in range(1):
        print(i)

if __name__ == '__main__':
    pool = Pool(os.cpu_count())
    print(os.cpu_count())
    outputs = pool.map(process_, 'occurence')



