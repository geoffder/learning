import numpy as np

w1 = 10.
w2 = .01
rate = .1

def w1_deriv(w1):
    return 2*w1

def w2_deriv(w2):
    return 4*w2**3

for i in range(1000):
    w1 -= rate * w1_deriv(w1)
    w2 -= rate * w2_deriv(w2)

    if ((i+1) % 100) == 0:
        print('w1:',w1,'w2', w2)
