import torch.cuda
import torchvision
#import tensorboard
from torch.optim import SGD
#from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from net import MyModel

#writer = SummaryWriter(log_dir='logs')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),  # Add this line
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#Get the dataset
#Training dataset
train_data_set = datasets.CIFAR10('./dataset',train=True,transform = transform,download=True)

# Test dataset
test_data_set = datasets.CIFAR10('./dataset',train=False,transform=transform,download=True)

# Size of the dataset
train_data_size = len(train_data_set)
test_data_size = len(train_data_set)

# Load the dataset
train_data_loader = DataLoader(train_data_set,batch_size=64,shuffle=True)
test_data_loader = DataLoader(test_data_set,batch_size=64,shuffle=True)

# Define the network
myModel = MyModel()

# Check if GPU is available
use_gpu = torch.cuda.is_available()
if(use_gpu):
    print('GPU is available')
    myModel = myModel.cuda()

# Number of epochs to train
epochs = 300

# Loss function
lossFn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = SGD(myModel.parameters(),lr=0.01)


for epoch in range(epochs):
    print('Training to{}/{}'.format(epoch+1,epochs))

    # Loss variables
    train_total_loss = 0.0
    test_total_loss = 0.0

    # Accuracy variables
    train_total_acc = 0.0
    test_total_acc = 0.0

    # Start training
    for data in train_data_loader:
        inputs,labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Clear gradients
        optimizer.zero_grad()

        outputs = myModel(inputs)

        # Calculate the loss by comparing the predicted output with the true labels
        loss = lossFn(outputs,labels)

        # Backpropagation
        loss.backward()

        # Update the parameters of each layer
        optimizer.step()

        # Calculate accuracy
        # Get the index of the predicted values with the highest probability
        _,index = torch.max(outputs,1)
        acc = torch.sum(index == labels).item()

        train_total_loss +=loss.item()
        train_total_acc += acc

    # Testing
    with torch.no_grad():
        for data in test_data_loader:
            inputs,labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = myModel(inputs)

            # Calculate the loss
            loss = lossFn(outputs, labels)

            # Calculate accuracy
            _, index = torch.max(outputs, 1)
            acc = torch.sum(index == labels).item()

            test_total_loss += loss.item()
            test_total_acc += acc


    print('train loss: {},acc:{} test loss:{},acc：{}'.format(train_total_loss,train_total_acc/train_data_size,test_total_loss,test_total_acc/test_data_size))

    #writer.add_scalar('loss/train',train_total_loss,epoch+1)
    #writer.add_scalar('acc/train',train_total_acc/train_data_size,epoch+1)
    #writer.add_scalar('loss/test',test_total_loss,epoch+1)
    #writer.add_scalar('acc/test',test_total_acc/test_data_size,epoch+1)

    if((epoch+1)%50==0):
        torch.save(myModel,'model/model_{}.pth'.format(epoch+1))

    torch.save(myModel,'model/model.pth')

'''
The code includes the following steps:

Data preprocessing:

Randomly flip the images horizontally and apply random crops with padding to the images.
Convert the images to tensors.
Normalize the pixel values of the images.
Load the CIFAR-10 dataset for training and testing using torchvision.datasets.CIFAR10.

Create data loaders for the training and testing datasets using torch.utils.data.DataLoader. 
This helps in efficiently loading the data in batches during training and testing.

Define the CNN model by importing MyModel from the net module. The model is instantiated as myModel.

Check if a GPU is available and move the model to the GPU if it is.

Set the number of epochs to train the model.

Define the loss function as cross-entropy loss (torch.nn.CrossEntropyLoss).

Define the optimizer as stochastic gradient descent (SGD) (torch.optim.SGD), 
which will update the parameters of the model during training.

Start the training loop over the specified number of epochs:

Initialize variables for tracking the total loss and accuracy during training and testing.
Iterate over the training data loader and perform the following steps:
Clear the gradients of the optimizer.
Forward pass the input images through the model.
Calculate the loss between the predicted outputs and the true labels.
Backpropagate the gradients through the model.
Update the parameters of the model using the optimizer.
Calculate the accuracy by comparing the predicted labels with the true labels.
Iterate over the testing data loader and perform the same steps as above, but without backpropagation and parameter updates.
Print the training and testing loss and accuracy for the current epoch.
Optionally save the model checkpoints every 50 epochs.
Save the final trained model.
'''