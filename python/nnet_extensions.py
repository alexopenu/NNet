from nnet import *





def splitList(list,l):
    return list[:l], list[l:]

# Takes one NNEt object and layer l, and returns two NNet objects, generated by cutting the original network
# after layer l. Note that the input layer is considered layer 0.
def splitNNet(nnet,l):
    if (l<1) or (l>nnet.numLayers):
            print("nothing to do")
            return(False)
    weights1, weights2 = splitList(nnet.weights,l)
    biases1, biases2 = splitList(nnet.biases,l)

    new_input_size = nnet.layerSizes[l+1]


    mins1 = nnet.mins
    maxs2 = nnet.maxes

    means1 = nnet.means
    ranges2 = nnet.ranges



    maxs1 = [0]*new_input_size  #Not sure!
    mins2 = [0]*new_input_size  #Not sure!
    means2 =[0]*new_input_size  #Not sure!
    ranges1 =[0]*new_input_size #Not sure!

    #NOTE that these choices may affect the evaluation!

    nnet1 = NNet(weights1,biases1,mins1,maxs1,means1,ranges1)
    nnet2 = NNet(weights2,biases2,mins2,maxs2,means2,ranges2)

    return nnet1,nnet2





# def evaluate_network(network, inputs):
#     '''
#     Evaluate network using given inputs
#
#     Args:
#         inputs (numpy array of floats): Network inputs to be evaluated
#
#     Returns:
#         (numpy array of floats): Network output
#     '''
#     numLayers = network.numLayers
#     inputSize = network.inputSize
#     outputSize = network.outputSize
#     biases = network.biases
#     weights = network.weights
#
#     # Prepare the inputs to the neural network
#     inputsNorm = np.zeros(inputSize)
#     for i in range(inputSize):
#         if inputs[i] < network.mins[i]:
#             inputsNorm[i] = (network.mins[i] - network.means[i]) / network.ranges[i]
#         elif inputs[i] > network.maxes[i]:
#             inputsNorm[i] = (network.maxes[i] - network.means[i]) / network.ranges[i]
#         else:
#             inputsNorm[i] = (inputs[i] - network.means[i]) / network.ranges[i]
#
#             # Evaluate the neural network
#     for layer in range(numLayers - 1):
#         inputsNorm = np.maximum(np.dot(weights[layer], inputsNorm) + biases[layer], 0)
#     outputs = np.dot(weights[-1], inputsNorm) + biases[-1]
#
#     # Undo output normalization
#     for i in range(outputSize):
#         outputs[i] = outputs[i] * network.ranges[-1] + network.means[-1]
#     return outputs
#
#
# def evaluate_network_multiple(self, inputs):
#     '''
#     Evaluate network using multiple sets of inputs
#
#     Args:
#         inputs (numpy array of floats): Array of network inputs to be evaluated.
#
#     Returns:
#         (numpy array of floats): Network outputs for each set of inputs
#     '''
#
#     numLayers = self.numLayers
#     inputSize = self.inputSize
#     outputSize = self.outputSize
#     biases = self.biases
#     weights = self.weights
#     inputs = np.array(inputs).T
#
#     # Prepare the inputs to the neural network
#     numInputs = inputs.shape[1]
#     inputsNorm = np.zeros((inputSize, numInputs))
#     for i in range(inputSize):
#         for j in range(numInputs):
#             if inputs[i, j] < self.mins[i]:
#                 inputsNorm[i, j] = (self.mins[i] - self.means[i]) / self.ranges[i]
#             elif inputs[i, j] > self.maxes[i]:
#                 inputsNorm[i, j] = (self.maxes[i] - self.means[i]) / self.ranges[i]
#             else:
#                 inputsNorm[i, j] = (inputs[i, j] - self.means[i]) / self.ranges[i]
#
#     # Evaluate the neural network
#     for layer in range(numLayers - 1):
#         inputsNorm = np.maximum(np.dot(weights[layer], inputsNorm) + biases[layer].reshape((len(biases[layer]), 1)),
#                                 0)
#     outputs = np.dot(weights[-1], inputsNorm) + biases[-1].reshape((len(biases[-1]), 1))
#
#     # Undo output normalization
#     for i in range(outputSize):
#         for j in range(numInputs):
#             outputs[i, j] = outputs[i, j] * self.ranges[-1] + self.means[-1]
#     return outputs.T
#
#
