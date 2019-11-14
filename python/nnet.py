import numpy as np
import sys
sys.path.append('../utils')
from writeNNet import writeNNet


def output_activation(a, b):
    return a


class NNet():
    """
    Class that represents a fully connected ReLU network in .nnet format
    
    Args:
        filename (str): A .nnet file to load

        or

        weights (list): Weight matrices in the network order
        biases (list): Bias vectors in the network order
        inputMins (list): Minimum values for each input
        inputMaxes (list): Maximum values for each input
        means (list): Mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
        ranges (list): Range values for each input and a range value for all outputs. Used to normalize inputs/outputs

    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        weights (list of numpy arrays): Weight matrices in network
        biases (list of numpy arrays): Bias vectors in network
    """

    def __init__(self, weights, biases, inputMinimums, inputMaximums, inputMeans, inputRanges, numLayers = -1, layerSizes = [], inputSize = -1, outputSize = -1):

        # Compute network parameters that can be computed from the rest
        _numLayers = len(weights)
        _inputSize = len(inputMinimums)
        _outputSize = len(biases[-1])

        # Find maximum size of any hidden layer
        _maxLayerSize = _inputSize
        for b in biases:
            if len(b) > _maxLayerSize:
                _maxLayerSize = len(b)
        #Create a list of layer Sizes
        _layerSizes = []
        _layerSizes.append(_inputSize)
        for b in biases:
            _layerSizes.append(len(b))



        if numLayers == -1:
            numLayers = _numLayers
        if inputSize == -1:
            inputSize = _inputSize
        if outputSize == -1:
            outputSize = _outputSize
        if layerSizes == []:
            layerSizes = _layerSizes


        #Checking that the parameters provided in the arguments makes what we have computed


        inputError = False
        if numLayers != _numLayers:
            numLayers = _numLayers
            inputError = True
            errorPlace = 1
        if inputSize != _inputSize:
            inputSize = _inputSize
            inputError = True
            errorPlace = 2
        if outputSize != _outputSize:
            outputSize = _outputSize
            inputError = True
            errorPlace = 3
        if layerSizes != _layerSizes:
            layerSizes = _layerSizes
            inputError = True
            errorPlace = 4


        if inputError:
            print("\nSomething was wrong with the arguments, corrected! Error code %d\n",errorPlace)


        self.numLayers = numLayers
        self.layerSizes = layerSizes
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.mins = inputMinimums
        self.maxes = inputMaximums
        self.means = inputMeans
        self.ranges = inputRanges
        self.weights = weights
        self.biases = biases



    @classmethod
    def fromfilename(cls, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line=f.readline()
                cnt+= 1
            #numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line=f.readline()

            #input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line=f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            weights=[]
            biases = []
            for layernum in range(numLayers):

                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum+1]
                weights.append([])
                biases.append([])
                weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
                for i in range(currentLayerSize):
                    line=f.readline()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]
                    for j in range(previousLayerSize):
                        weights[layernum][i,j] = aux[j]
                #biases
                biases[layernum] = np.zeros(currentLayerSize)
                for i in range(currentLayerSize):
                    line=f.readline()
                    x = float(line.strip().split(",")[0])
                    biases[layernum][i] = x
            '''
            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges
            self.weights = weights
            self.biases = biases
            '''

            return cls(weights, biases, inputMinimums, inputMaximums, inputMeans, inputRanges, numLayers, layerSizes, inputSize, outputSize)


    def evaluate_network(self, inputs, normalize_inputs = True, normalize_outputs = True, activate_output_layer = False):

        '''
        Evaluate network using given inputs

        Args:
            inputs (numpy array of floats): Network inputs to be evaluated
            
        Returns:
            (numpy array of floats): Network output
        '''
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights

        # Prepare the inputs to the neural network
        if (normalize_inputs):
            inputsNorm = np.zeros(inputSize)
            for i in range(inputSize):
                if inputs[i]<self.mins[i]:
                    inputsNorm[i] = (self.mins[i]-self.means[i])/self.ranges[i]
                elif inputs[i]>self.maxes[i]:
                    inputsNorm[i] = (self.maxes[i]-self.means[i])/self.ranges[i]
                else:
                    inputsNorm[i] = (inputs[i]-self.means[i])/self.ranges[i]
        else:
            inputsNorm = inputs



        # Evaluate the neural network
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer],0)

        if (activate_output_layer):
            outputs = np.maximum(np.dot(weights[-1],inputsNorm)+biases[-1],0)
        else:
            outputs = np.dot(weights[-1],inputsNorm)+biases[-1]

        # Undo output normalization
        if (normalize_outputs):
            for i in range(outputSize):
                outputs[i] = outputs[i]*self.ranges[-1]+self.means[-1]

        return outputs

    def evaluate_network_multiple(self,inputs, normalize_inputs = True, normalize_outputs = True, activate_output_layer = False):
        '''
        Evaluate network using multiple sets of inputs
        
        Args:
            inputs (numpy array of floats): Array of network inputs to be evaluated.
            
        Returns:
            (numpy array of floats): Network outputs for each set of inputs
        '''
        
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights
        inputs = np.array(inputs).T

        # Prepare the inputs to the neural network
        numInputs = inputs.shape[1]

        if (normalize_inputs):
            inputsNorm = np.zeros((inputSize,numInputs))
            for i in range(inputSize):
                for j in range(numInputs):
                    if inputs[i,j]<self.mins[i]:
                        inputsNorm[i,j] = (self.mins[i]-self.means[i])/self.ranges[i]
                    elif inputs[i,j] > self.maxes[i]:
                        inputsNorm[i,j] = (self.maxes[i]-self.means[i])/self.ranges[i]
                    else:
                        inputsNorm[i,j] = (inputs[i,j]-self.means[i])/self.ranges[i]
        else:
            inputsNorm = inputs


        # Evaluate the neural network
        for layer in range(numLayers-1):
            inputsNorm = np.maximum(np.dot(weights[layer],inputsNorm)+biases[layer].reshape((len(biases[layer]),1)),0)

        if (activate_output_layer):
            outputs = np.maximum(np.dot(weights[-1], inputsNorm) + biases[-1].reshape((len(biases[-1]), 1)),0)
        else:
            outputs = np.dot(weights[-1],inputsNorm)+biases[-1].reshape((len(biases[-1]),1))

        # Undo output normalization
        if (normalize_outputs):
            for i in range(outputSize):
                for j in range(numInputs):
                    outputs[i,j] = outputs[i,j]*self.ranges[-1]+self.means[-1]

        return outputs.T

    def evaluate_network_nonorm(self, inputs):

        '''
        Evaluate network using given inputs, without normalizing the inputs

        Args:
            inputs (numpy array of floats): Network inputs to be evaluated

        Returns:
            (numpy array of floats): Network (INCLUDING a ReLU output layer) output
        '''
        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights

        # # Prepare the inputs to the neural network
        # inputsNorm = np.zeros(inputSize)
        # for i in range(inputSize):
        #     if inputs[i] < self.mins[i]:
        #         inputsNorm[i] = (self.mins[i] - self.means[i]) / self.ranges[i]
        #     elif inputs[i] > self.maxes[i]:
        #         inputsNorm[i] = (self.maxes[i] - self.means[i]) / self.ranges[i]
        #     else:
        #         inputsNorm[i] = (inputs[i] - self.means[i]) / self.ranges[i]


        #No normalization
        inputsNorm = inputs

        # Evaluate the neural network
        for layer in range(numLayers - 1):
            inputsNorm = np.maximum(np.dot(weights[layer], inputsNorm) + biases[layer], 0)
        outputs = output_activation(np.dot(weights[-1], inputsNorm) + biases[-1], 0)

        # # Undo output normalization
        # for i in range(outputSize):
        #     outputs[i] = outputs[i] * self.ranges[-1] + self.means[-1]
        return outputs

    def evaluate_network_multiple_nonorm(self, inputs):
        '''
        Evaluate network using multiple sets of inputs without normalization

        Args:
            inputs (numpy array of floats): Array of network inputs to be evaluated.

        Returns:
            (numpy array of floats): Network (INCLUDING a ReLU output layer) outputs for each set of inputs
        '''

        numLayers = self.numLayers
        inputSize = self.inputSize
        outputSize = self.outputSize
        biases = self.biases
        weights = self.weights
        inputs = np.array(inputs).T

        # Prepare the inputs to the neural network
        numInputs = inputs.shape[1]
        # inputsNorm = np.zeros((inputSize, numInputs))
        # for i in range(inputSize):
        #     for j in range(numInputs):
        #         if inputs[i, j] < self.mins[i]:
        #             inputsNorm[i, j] = (self.mins[i] - self.means[i]) / self.ranges[i]
        #         elif inputs[i, j] > self.maxes[i]:
        #             inputsNorm[i, j] = (self.maxes[i] - self.means[i]) / self.ranges[i]
        #         else:
        #             inputsNorm[i, j] = (inputs[i, j] - self.means[i]) / self.ranges[i]



        inputsNorm = inputs

        # Evaluate the neural network
        for layer in range(numLayers - 1):
            inputsNorm = np.maximum(np.dot(weights[layer], inputsNorm) + biases[layer].reshape((len(biases[layer]), 1)),
                                           0)
        outputs = output_activation(np.dot(weights[-1], inputsNorm) + biases[-1].reshape((len(biases[-1]), 1)),0)

        # # Undo output normalization
        # for i in range(outputSize):
        #     for j in range(numInputs):
        #         outputs[i, j] = outputs[i, j] * self.ranges[-1] + self.means[-1]

        return outputs.T

    def num_inputs(self):
        ''' Get network input size'''
        return self.inputSize

    def num_outputs(self):
        ''' Get network output size'''
        return self.outputSize

    def write_to_file(self,fileName):
        '''write network into a file'''
        writeNNet(self.weights, self.biases, self.mins, self.maxes, self.means, self.ranges, fileName)