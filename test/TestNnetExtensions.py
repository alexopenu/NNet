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

nnet = NNet(weights, biases, inputMins, inputMaxes, means, ranges)
nnet.write_to_file(fileTarget)


print(filecmp.cmp(fileOrigin,fileTarget)) #True

writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,fileTarget2)
print(filecmp.cmp(fileOrigin,fileTarget2)) #True

writeNNet(nnet.weights,nnet.biases,nnet.mins,nnet.maxes,nnet.means,nnet.ranges,fileTarget2)
print(filecmp.cmp(fileOrigin,fileTarget2)) #True

'''
print nnet2.numLayers
x = splitNNet(nnet2,5)
print x
'''


nnet = NNet.fromfilename(fileOrigin)
#nnet.write_to_file(fileTarget2)
'''
writeNNet(nnet.weights, nnet.biases, nnet.mins, nnet.maxes, nnet.means, nnet.ranges, fileTarget2) #Error!
print(filecmp.cmp(fileOrigin,fileTarget2))
'''

half = nnet.numLayers/2


list = [1,2,3,4]

list1,list2 = splitList(list,2)
print(list1)
print(list2)

nnet_1,nnet_2 = splitNNet(nnet,half)

print nnet.numLayers
print nnet_1.numLayers
print nnet_2.numLayers



outputs=nnet.evaluate_network_nonorm([15299.0, 0.0, -3.1, 600.0, 500.0])
print(outputs)


outputs1 = nnet_1.evaluate_network_nonorm([15299.0, 0.0, -3.1, 600.0, 500.0])
print(len(outputs1))
#print(outputs1)
outputs2 = nnet_2.evaluate_network_nonorm(outputs1)
print(outputs2)

print(outputs == outputs2)

'''
print(len(outputs) == len(outputs2))
print len(outputs)

x = [1.0,2.0]
y = [1.0,4.0]

print x==y

x = [1,2]
y = [1,2]
print x==y
y.append(3)
print(y)
print(x)
print x==y
y=x.append(3)
print(y)
print(x)
print x==y
'''











