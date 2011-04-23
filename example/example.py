"""
Run IBP on the synthetic 'Cambridge Bars' dataset
"""
import cPickle as CP
import numpy as NP

from PyIBP import PyIBP as IBP

# Define hyperparameters
(alpha, alpha_a, alpha_b) = (1., 1., 1.)
(sigma_x, sx_a, sx_b) = (1., 1., 1.)
(sigma_a, sa_a, sa_b) = (1., 1., 1.)

numsamp = 2000

# Load and center the data
(trueWeights,features,data) = CP.load(open('block_image_set.p'))
(N,D) = data.shape
cdata = IBP.centerData(data)

# Initialize the model
f = IBP(cdata,(alpha,alpha_a,alpha_b),
        (sigma_x, sx_a, sx_b),
        (sigma_a, sa_a, sa_b))

# Do inference
for s in range(numsamp):
    # Print current chain state
    f.sampleReport(s)
    print 'Learned weights (rounded)'
    for factor in NP.round(f.weights()).astype(NP.int):
        print str(factor.reshape((6,6)))
    print 'True weights'
    for factor in trueWeights:
        print str(factor.reshape((6,6)))    
    # Take a new sample
    f.fullSample()    
