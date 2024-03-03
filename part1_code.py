import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import gdown

def positional_encoding(x, num_frequencies=6, incl_input=True):

    """
    Apply positional encoding to the input.

    Args:
    x (torch.Tensor): Input tensor to be positionally encoded.
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # print("x",x.shape)

    results = []
    if incl_input:
        results.append(x)
    # encode input tensor and append the encoded tensor to the list of results.
    for i in range(num_frequencies):
      results.append(torch.sin(2**i * np.pi * x))
      results.append(torch.cos(2**i * np.pi * x))
    # print("result:",len(results),len(results[0]))
    return torch.cat(results, dim=-1)

class model_2d(nn.Module):

    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """

    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        self.layer_in = nn.Linear(2+num_frequencies*2*2, filter_size)
        self.layer = nn.Linear(filter_size, filter_size)
        self.layer_out = nn.Linear(filter_size, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # example of forward through a layer: y = self.layer_in(x)
        # print("forward input:",x.shape)
        y = self.layer_in(x)
        y = self.relu(y)
        y = self.layer(y)
        y = self.relu(y)
        y = self.layer_out(y)
        y = self.sigmoid(y)
        # print("out:",y.shape)
        return y

def normalize_coord(height, width, num_frequencies=6):

    """
    Creates the 2D normalized coordinates, and applies positional encoding to them

    Args:
    height (int): Height of the image
    width (int): Width of the image
    num_frequencies (optional, int): The number of frequencies used in
      the positional encoding (default: 6).

    Returns:
    (torch.Tensor): Returns the 2D normalized coordinates after applying positional encoding to them.
    """

    # Create the 2D normalized coordinates, and apply positional encoding to them
    normalized_coordinates = []
    xx=torch.linspace(0, 1, width)
    yy=torch.linspace(0, 1, height)
    x,y = torch.meshgrid(xx,yy,indexing='ij')
    normalized_coordinates = torch.stack([x, y], dim=-1)
    # print("normalized_coordinates.shape",normalized_coordinates.shape)
    normalized_coordinates=normalized_coordinates.reshape(-1,2)
    embedded_coordinates = positional_encoding(normalized_coordinates, num_frequencies=num_frequencies)
    # print(embedded_coordinates.shape)
    return embedded_coordinates


def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding, show=True):

    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000

    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)

    # Define the optimizer
    optimizer=torch.optim.Adam(model2d.parameters(), lr=lr)

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()
    # Create the 2D normalized coordinates, and apply positional encoding to them
    normalized_coordinates = normalize_coord(height, width, num_frequencies=num_frequencies)
    normalized_coordinates=normalized_coordinates.to(device)

    for i in range(iterations+1):
        optimizer.zero_grad()
        # Run one iteration
        model2d.train()
        # print("input:",normalized_coordinates.shape)
        pred = model2d(normalized_coordinates)
        # Compute mean-squared error between the predicted and target images. Backprop!
        loss = torch.mean((pred.view(-1,3)-test_img.view(-1, 3))**2)
        pred=pred.view(height, width, 3)
        loss.backward()
        optimizer.step()

        # Display images/plots/stats
        if i % display == 0 and show:
            # Calculate psnr
            psnr = 10 * torch.log10(1 / loss)

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

    print('Done!')
    print(num_frequencies)
    torch.save(model2d.state_dict(),'model_2d_' + str(num_frequencies) + 'freq.pt')
    plt.imsave('van_gogh_' + str(num_frequencies) + 'freq.png',pred.detach().cpu().numpy())
    return pred.detach().cpu()

