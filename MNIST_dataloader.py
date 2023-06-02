# %% imports
# pytorch
import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
import numpy as np
# pyplot
import matplotlib.pyplot as plt

# %% Noisy MNIST dataset
class Noisy_MNIST(Dataset):
    # initialization of the dataset
    def __init__(self, split,data_loc,noise=0.5):
        # save the input parameters
        self.split    = split 
        self.data_loc = data_loc
        self.noise    = noise
        
        if self.split == 'train':
            Train = True
        else:
            Train = False
            
        # get the original MNIST dataset   
        Clean_MNIST = datasets.MNIST(self.data_loc, train=Train, download=True)
        
        # reshuffle the test set to have digits 0-9 at the start
        if self.split == 'train':
            data = Clean_MNIST.data.unsqueeze(1)
        else:
            data = Clean_MNIST.data.unsqueeze(1)
            idx = torch.load('test_idx.tar')
            data[:,:] = data[idx,:]
            
        
        # reshape and normalize
        resizer = transforms.Resize(32)
        resized_data = resizer(data)*1.0
        normalized_data = 2 *(resized_data/255) - 1
        #normalized_data = (resized_data - 33)/74
        
        # create the data
        self.Clean_Images = normalized_data
        self.Noisy_Images = normalized_data + torch.randn(normalized_data.size())*self.noise
        self.Labels       = Clean_MNIST.targets
    
    # return the number of examples in this dataset
    def __len__(self):
        return self.Labels.size(0)
    
    # create a a method that retrieves a single item form the dataset
    def __getitem__(self, idx):
        clean_image = self.Clean_Images[idx,:,:,:]
        noisy_image = self.Noisy_Images[idx,:,:,:]
        label =  self.Labels[idx]
        
        return clean_image,noisy_image,label
    
# %% dataloader for the Noisy MNIST dataset
def create_dataloaders(data_loc, batch_size):
    Noisy_MNIST_train = Noisy_MNIST("train", data_loc)
    Noisy_MNIST_test  = Noisy_MNIST("test" , data_loc)
    
    Noisy_MNIST_train_loader =  DataLoader(Noisy_MNIST_train, batch_size=batch_size, shuffle=True,  drop_last=False)
    Noisy_MNIST_test_loader  =  DataLoader(Noisy_MNIST_test , batch_size=batch_size, shuffle=False, drop_last=False)
    
    return Noisy_MNIST_train_loader, Noisy_MNIST_test_loader

# %% test if the dataloaders work
if __name__ == "__main__":
    # define parameters
    data_loc = 'D://5LSL0-Datasets' #change the datalocation to something that works for you
    batch_size = 64
    
    # get dataloader
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    
    # get some examples
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    # use these example images througout the assignment as the first 10 correspond to the digits 0-9
    
    # show the examples in a plot
    plt.figure(figsize=(12,3))
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(2,10,i+11)
        plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("data_examples.png",dpi=300,bbox_inches='tight')
    plt.show()
# %%
# Week 1
# 1A create python function for denoising MNIST

# ISTA function for denoising
def ista_denoise(noisy_image, step_size, num_steps, lambda_val):
    # Define the denoising network
    denoiser = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 1, kernel_size=3, padding=1)
    )

    # Scale the noisy image to the range [0, 2] and shift it to the range [-1, 1]
    noisy_image = (noisy_image + 1) * 0.5

    # Initialize the denoised image
    denoised_image = torch.zeros_like(noisy_image)

    # Perform ISTA denoising
    for i in range(num_steps):
        # Compute the gradient of the denoising network
        gradient = denoiser(denoised_image)

        # Compute the residual
        residual = noisy_image - gradient

        # Update the denoised image
        denoised_image = denoised_image + step_size * residual

        # Apply proximal operator (soft-thresholding)
        denoised_image = soft_threshold(denoised_image, lambda_val * step_size)

    # Scale the denoised image back to the range [-1, 1] and shift it to the range [0, 1]
    denoised_image = denoised_image * 2 - 1
    denoised_image = torch.clamp(denoised_image, 0, 1)

    # Convert the denoised image back to a numpy array
    denoised_image = denoised_image.squeeze().squeeze().detach().numpy()

    return denoised_image

def soft_threshold(x, threshold):
    return torch.sign(x) * torch.max(torch.abs(x) - threshold, torch.zeros_like(x))


# %%

data_loc = 'D://5LSL0-Datasets' #change the datalocation to something that works for you
batch_size = 10
    
# get dataloader
train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    
# get some examples
examples = enumerate(test_loader)
_, (x_clean_example, x_noisy_example, labels_example) = next(examples)

# %% 1B Show results of denoising
# Define constants
mu = 0.1
K = 100
lambda_test = 0.1

# Create empty lists
denoised_images_list = []
clear_images_list = []
noisy_images_list = []

for i in range(batch_size):
    denoised_image = ista_denoise(x_noisy_example[i,:,:,:], step_size=mu, num_steps=K, lambda_val=lambda_test)
    denoised_images_list.append(denoised_image)

    clear_images_list.append(x_clean_example[i,:,:,:])

    noisy_images_list.append(x_noisy_example[i,:,:,:])

# %%
# Assuming you have a list of noisy images and denoised images
clear_images = clear_images_list  # List of clear images (NumPy arrays)
denoised_images = denoised_images_list  # List of denoised images (NumPy arrays)
noisy_images = noisy_images_list # List of noisy images (NumPy arrays)

num_images = len(denoised_images)

# Create a figure with subplots for each image
fig, axes = plt.subplots(nrows=3, ncols=num_images, figsize=(6 * num_images, 8))

for i in range(num_images):
    # Get the noisy image and denoised image for the current iteration
    noisy_images_np = noisy_images [i].squeeze()
    clear_image_np = clear_images[i].squeeze()
    denoised_image_np = denoised_images[i].squeeze()

    # Plot the denoised image in the first row of subplots
    axes[0, i].imshow(noisy_images_np, cmap='gray')
    axes[0, i].set_title('Noisy Image')
    axes[0, i].axis('off')  

    # Plot the denoised image in the first row of subplots
    axes[1, i].imshow(denoised_image_np, cmap='gray')
    axes[1, i].set_title('Denoised Image')
    axes[1, i].axis('off')  

    # Plot the clear image in the second row of subplots
    axes[2, i].imshow(clear_image_np, cmap='gray')
    axes[2, i].set_title('Clear Image')
    axes[2, i].axis('off')


# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()

# Save Figure

plt.savefig('1B_denoised_ISTA')
# %% 1C Evalute performance
