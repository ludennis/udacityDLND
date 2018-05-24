import time
from collections import namedtuple

import numpy as np
import tensorflow as tf


def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = batch_size * n_steps
    n_batches = int(len(arr) / characters_per_batch)
    
    # Keep only enough characters to make full batches
    arr = arr[:batch_size * n_steps * n_batches]
    
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size,int(len(arr)/batch_size)))
    
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:,n*n_steps:n*(n_steps+1)]
        # The targets, shifted by one
        y = arr[:,n*n_steps+1:n*(n_steps+1)+1]
        yield x, y


if __name__ == '__main__':
	with open('anna.txt', 'r') as f:
	    text=f.read()
	vocab = sorted(set(text))
	vocab_to_int = {c: i for i, c in enumerate(vocab)}
	int_to_vocab = dict(enumerate(vocab))
	encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

	batches = get_batches(encoded, 10, 50)
	x, y = next(batches)

	print('x\n', x[:10, :10])
	print('\ny\n', y[:10, :10])
