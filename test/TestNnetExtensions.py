import sys
sys.path.append('../python')
sys.path.append('../utils')

import filecmp

from nnet import *
from nnet_extensions import *
from readNNet import readNNet


fileOrigin = '../nnet/TestNetwork.nnet'
fileTarget = '../nnet/TestNetwork_output.nnet'
fileTarget2 = '../nnet/TestNetwork_output2.nnet'


(weights, biases, inputMins, inputMaxes, means, ranges) = readNNet(fileOrigin,True)

nnet2 = NNet(weights, biases, inputMins, inputMaxes, means, ranges)
nnet2.write_to_file(fileTarget)


print(filecmp.cmp(fileOrigin,fileTarget)) #True

writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,fileTarget2)
print(filecmp.cmp(fileOrigin,fileTarget2)) #True

writeNNet(nnet2.weights,nnet2.biases,nnet2.mins,nnet2.maxes,nnet2.means,nnet2.ranges,fileTarget2)
print(filecmp.cmp(fileOrigin,fileTarget2)) #True

'''
print nnet2.numLayers
x = splitNNet(nnet2,5)
print x
'''
nnet = NNet.fromfilename(fileOrigin)
nnet.write_to_file(fileTarget2)
#writeNNet(nnet.weights, nnet.biases, nnet.mins, nnet.maxes, nnet.means, nnet.ranges, fileTarget2)
print(filecmp.cmp(fileOrigin,fileTarget2))














