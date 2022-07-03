import torch.nn as nn
import torch.nn.init

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, list_hidden, num_classes):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.list_hidden = list_hidden
        self.num_classes = num_classes

        self.create_network()
        self.init_weights()

    def create_network(self):
        layers = []

        # layers list with correct values for parameters in_features and
        # out_features. This is the first layer of the network.
        layers.append(nn.Linear(in_features=self.input_size,
                      out_features=self.list_hidden[0]))

        # adding a ReLU activation function
        layers.append(nn.ReLU(inplace=True))

        # Iterate over other hidden layers just before the last layer
        for i in range(len(self.list_hidden) - 1):
            # the layers list according to the values in self.list_hidden.
            layers.append(
                nn.Linear(self.list_hidden[i], self.list_hidden[i + 1]))

            # adding the ReLU activation function
            layers.append(nn.ReLU(inplace=True))
        # layers list with correct values for parameters in_features and
        # out_features. This is the last layer of the network.
        layers.append(
            nn.Linear(in_features=self.list_hidden[-1], out_features=self.num_classes))

        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def init_weights(self):
        """Initializes the weights of the network. Weights of a
        torch.nn.Linear layer should be initialized from a normal
        distribution with mean 0 and standard deviation 0.1. Bias terms of a
        torch.nn.Linear layer should be initialized with a constant value of 0.
        """
        torch.manual_seed(0)

        # For each layer in the network
        for module in self.modules():
            # If it is a torch.nn.Linear layer
            if isinstance(module, nn.Linear):
                # from a normal distribution with mean 0 and standard deviation
                # of 0.1.
                nn.init.normal_(module.weight, std=0.1)

                # with a constant value of 0.
                nn.init.constant_(module.bias, 0)

    def forward(self,
                x,
                verbose=False):
        """Forward propagation of the model, implemented using PyTorch.

        Arguments:
            x {torch.Tensor} -- A Tensor of shape (N, D) representing input
            features to the model.
            verbose {bool, optional} -- Indicates if the function prints the
            output or not.

        Returns:
            torch.Tensor, torch.Tensor -- A Tensor of shape (N, C) representing
            the output of the final linear layer in the network. A Tensor of
            shape (N, C) representing the probabilities of each class given by
            the softmax function.
        """

        # For each layer in the network
        for i in range(len(self.layers) - 1):

            # Call the forward() function of the layer
            # and return the result to x.
            x = self.layers[i](x)

            if verbose:
                # Print the output of the layer
                print('Output of layer ' + str(i))
                print(x, '\n')

        # Apply the softmax function
        probabilities = self.layers[-1](x)

        if verbose:
            print('Output of layer ' + str(len(self.layers) - 1))
            print(probabilities, '\n')

        return x, probabilities

    def predict(self,
                probabilities):
        """Returns the index of the class with the highest probability.

        Arguments:
            probabilities {torch.Tensor} -- A Tensor of shape (N, C)
            representing the probabilities of N instances for C classes.

        Returns:
            torch.Tensor -- A Tensor of shape (N, ) contaning the indices of
            the class with the highest probability for N instances.
        """

        return torch.argmax(probabilities, dim=1)
