import torch
from torch import nn

#Define a class that inherits from nn.Module as a model
class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The layers that define the model
        # The first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # First maximum pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Second maximum pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Third convolution layer
        self.conv3 = nn.Conv2d(32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # Third maximum pooling layer
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # Flattening features into one-dimensional vectors
        self.flatten = nn.Flatten()
        # Fully connected layer 1
        self.linear1 = nn.Linear(1024, 64)
        # Fully connected layer 2
        self.linear2 = nn.Linear(64, 10)
        # Classification using softmax function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward propagation function
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

# Create input data
x = torch.randn(1, 3, 32, 32)

# Create model instances
myModel = MyModel()

# Perform forward propagation calculations
out = myModel(x)

print(out)
'''
The CNN model consists of the following layers:

Convolutional Layers (nn.Conv2d): There are three convolutional layers that are used to extract features from the input images. 
The first convolutional layer has an input channel of 3, an output channel of 32, a kernel size of 5x5, a stride of 1, 
and a padding of 2. The next two convolutional layers have an input channel of 32 and output channels of 32 and 64 respectively. 
Other parameters are the same as the first convolutional layer.

Convolutional layers are responsible for extracting features from the input image. Each convolutional layer applies a 
set of learnable filters to the input image, producing feature maps that capture different patterns and characteristics. 
The number of output channels determines the number of filters used, which helps the model learn a variety of features 
at different levels of abstraction.

Max Pooling Layers (nn.MaxPool2d): There are three max pooling layers that perform downsampling. Each max pooling layer 
has a kernel size of 2x2 and performs a 2x downsampling on the feature maps.

Max pooling layers perform downsampling by dividing the input feature maps into non-overlapping regions and taking the 
maximum value within each region. This downsampling reduces the spatial dimensions of the feature maps while retaining 
the most salient features. Max pooling helps the model become more invariant to small translations and reduces the number of parameters.

Flatten Layer (nn.Flatten): This layer flattens the feature maps into a 1D vector to be input to the fully connected layers.

The flattening layer transforms the multidimensional feature maps into a one-dimensional vector. It concatenates all the 
values in the feature maps into a long vector, which serves as the input to the fully connected layers.

Fully Connected Layers (nn.Linear): There are two fully connected layers that map the flattened features to the final 
classification output. The first fully connected layer maps an input dimension of 1024 to a dimension of 64, 
and the second fully connected layer maps a dimension of 64 to a dimension of 10, representing the 10 classes for classification.

Fully connected layers, also known as dense layers, are responsible for learning complex patterns and relationships in 
the extracted features. Each neuron in the fully connected layer is connected to all the neurons in the previous layer. 
The output of the fully connected layers gradually transforms the high-level features into class probabilities or scores.

Softmax Activation (nn.Softmax): The softmax function is applied to the final output to obtain a probability distribution over the classes.
The softmax activation function is applied to the output of the last fully connected layer. It converts the raw output 
values into a probability distribution over the classes. Each element in the output vector represents the probability of 
the input image belonging to the corresponding class. The softmax function ensures that the probabilities sum up to 1, 
making it suitable for multi-class classification tasks.
'''