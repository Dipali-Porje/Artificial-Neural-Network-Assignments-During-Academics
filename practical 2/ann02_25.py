
#second practical

import numpy as np

def mp_neuron(inputs, weights, threshold):
  weighted_sum = np.dot(inputs, weights)
  output = 1 if weighted_sum >= threshold else 0
  return output


def and_not(x1,x2):
  weights = (1,-1)
  threshold = 1
  inputs = np.array([x1,x2])
  output = mp_neuron(inputs, weights, threshold)
  return output

print('input : 0,0\n Output is :',and_not(0,0))
print('\ninput : 1,0\n Output is :',and_not(1,0))
print('\ninput : 0,1\n Output is :',and_not(0,1))
print('\ninput : 1,1\n Output is :',and_not(1,1))
