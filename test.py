import os
import cv2
import torchvision.transforms as transforms
import torch

#Define the class labels for the model
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Check if GPU is available
use_gpu = torch.cuda.is_available()

# Load the pre-trained model
model = torch.load('./model/model_300.pth', map_location=torch.device('cuda' if use_gpu else 'cpu'))

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Specify the folder path where test images are located
folder_path = './testimages'

# Get a list of file names in the folder
files = os.listdir(folder_path)

# Create a list of image file paths
image_files = [os.path.join(folder_path, f) for f in files]

# Iterate over each image file
for img in image_files:
    # Read the image using OpenCV
    image = cv2.imread(img)
    cv2.imshow('image', image)
    # Resize the image to (32, 32) using OpenCV
    image = cv2.resize(image, (32, 32))
    # Convert the image color space from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply image transformations
    image = transform(image)
    image = image.unsqueeze(0)  # Add this line to add a batch dimension
    # Move the image tensor to GPU if available
    image = image.to('cuda' if use_gpu else 'cpu')
    output = model(image)

    # Get the predicted class label
    _, index = torch.max(output, 1)
    pre_val = classes[index]

    print('Predicted class:', pre_val)

    # Wait for key press to display the next image
    cv2.waitKey(0)
