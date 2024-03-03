"""### Part 2: Fitting a 3D Image"""

import os
import gdown
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

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

"""2.1 Complete the following function that calculates the rays that pass through all the pixels of an HxW image"""

def get_rays(height, width, intrinsics, w_R_c, w_T_c):

    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
    w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    # Adjust the shape of w_T_c for broadcasting
    w_T_c_broadcasted = w_T_c.reshape(1, 1, 3)

    # Broadcast w_T_c to the desired shape
    ray_origins = torch.broadcast_to(w_T_c_broadcasted, (height, width, 3))

    u0=intrinsics[0][2]
    v0=intrinsics[1][2]
    fx=intrinsics[0][0]
    fy=intrinsics[1][1]
    # create tensor torch.Tensor
    yy = torch.arange(height).to(device)
    h = (yy - v0) / fx
    xx = torch.arange(width).to(device)
    w = (xx - u0) / fy
    y, x = torch.meshgrid(h, w)

    # find dirs
    dirs = torch.stack([x, y, torch.ones_like(x)], -1)
    dirs = dirs.permute(2, 0, 1)
    dirs = dirs.reshape(3, -1)

    # Multiply Rcw matrix by the dirs tensor then sawp and reshape
    ray_directions=(torch.matmul(w_R_c, dirs)).permute(1, 0)
    ray_directions = ray_directions.reshape(width, height, 3)
    return ray_origins, ray_directions

"""Complete the next function to visualize how is the dataset created. You will be able to see from which point of view each image has been captured for the 3D object. What we want to achieve here, is to being able to interpolate between these given views and synthesize new realistic views of the 3D object."""

def plot_all_poses(poses):

    num_poses = poses.shape[0]
    poses = torch.tensor(poses, dtype=torch.float32)

    translations = poses[:, :3, 3]

    origins = translations
    directions = poses[:, :3, :3] @ torch.tensor([0.0, 0.0, 1.0], device=poses.device)
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[..., 0].flatten(),
                  origins[..., 1].flatten(),
                  origins[..., 2].flatten(),
                  directions[..., 0].flatten(),
                  directions[..., 1].flatten(),
                  directions[..., 2].flatten(), length=0.12, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.show()

"""2.2 Complete the following function to implement the sampling of points along a given ray."""

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.

    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    device=ray_origins.device
    height, width, _ = ray_origins.shape
    bounding=far-near
    points=torch.arange(samples, dtype=torch.float32)/(samples-1)
    depth_points=near+points*bounding
    depth_points=torch.broadcast_to(depth_points, (height, width, samples)).to(device)
    ray_points=ray_origins.reshape(height, width, 1, 3)+depth_points.reshape((height, width, samples, 1))*ray_directions.reshape(height, width, 1, 3)
    return ray_points, depth_points


"""2.3 Define the network architecture of NeRF along with a function that divided data into chunks to avoid memory leaks during training."""

class nerf_model(nn.Module):

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()
        # for autograder compliance, please follow the given naming for your layers
        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(num_x_frequencies*2*3 + 3,filter_size),
            'layer_2': nn.Linear(filter_size, filter_size),
            'layer_3': nn.Linear(filter_size, filter_size),
            'layer_4': nn.Linear(filter_size, filter_size),
            'layer_5': nn.Linear(filter_size, filter_size),
            'layer_6': nn.Linear(filter_size+num_x_frequencies*2*3 + 3, filter_size),
            'layer_7': nn.Linear(filter_size, filter_size),
            'layer_8': nn.Linear(filter_size, filter_size),
            'layer_s': nn.Linear(filter_size, 1),
            'layer_9': nn.Linear(filter_size, filter_size),
            'layer_10': nn.Linear(filter_size+num_d_frequencies*2*3 + 3, 128),
            'layer_11': nn.Linear(128, 3),
        })
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, d):
        print("forward start")
        # example of forward through a layer: y = self.layers['layer_1'](x)
        h1 =self.relu(self.layers['layer_1'](x))
        h2 =self.relu(self.layers['layer_2'](h1))
        h3 =self.relu(self.layers['layer_3'](h2))
        h4 =self.relu(self.layers['layer_4'](h3))
        h5 =self.relu(self.layers['layer_5'](h4))

        h6 =self.relu(self.layers['layer_6'](torch.cat([h5, x], dim=-1)))
        h7 =self.relu(self.layers['layer_7'](h6))
        h8 =self.relu(self.layers['layer_8'](h7))

        h9 =self.layers['layer_9'](h8)
        sigma = self.layers['layer_s'](h8)
        h10 = self.relu(self.layers['layer_10'](torch.cat([h9, d], dim=-1)))
        h11 = self.sigmoid(self.layers['layer_11'](h10))
        rgb = h11

        print("forward end")
        return rgb, sigma

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()
        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(num_x_frequencies*2*3 + 3,filter_size),
            'layer_2': nn.Linear(filter_size, filter_size),
            'layer_3': nn.Linear(filter_size, filter_size),
            'layer_4': nn.Linear(filter_size, filter_size),
            'layer_5': nn.Linear(filter_size, filter_size),
            'layer_6': nn.Linear(filter_size+num_x_frequencies*2*3 + 3, filter_size),
            'layer_7': nn.Linear(filter_size, filter_size),
            'layer_8': nn.Linear(filter_size, filter_size),
            'layer_s': nn.Linear(filter_size, 1),
            'layer_9': nn.Linear(filter_size, filter_size),
            'layer_10': nn.Linear(filter_size+num_d_frequencies*2*3 + 3, 128),
            'layer_11': nn.Linear(128, 3),
        })
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, d):
        # example of forward through a layer: y = self.layers['layer_1'](x)
        h1 =self.relu(self.layers['layer_1'](x))
        h2 =self.relu(self.layers['layer_2'](h1))
        h3 =self.relu(self.layers['layer_3'](h2))
        h4 =self.relu(self.layers['layer_4'](h3))
        h5 =self.relu(self.layers['layer_5'](h4))
        h6 =self.relu(self.layers['layer_6'](torch.cat([self.relu(h5), x], dim=-1)))
        h7 =self.relu(self.layers['layer_7'](h6))
        h8 =self.relu(self.layers['layer_8'](h7))
        h9 =self.layers['layer_9'](h8)
        sigma = self.layers['layer_s'](h8)
        h10 = self.relu(self.layers['layer_10'](torch.cat([h9, d], dim=-1)))
        h11 = self.sigmoid(self.layers['layer_11'](h10))
        rgb = h11
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    def get_chunks(inputs, chunksize = 2**15):
        """
        This fuction gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    # flatten the vector, apply positional encoding to it
    H,W,S,C=ray_points.shape
    ray_points_flatten=ray_points.reshape(-1, C)
    en_ray_points = positional_encoding(ray_points_flatten, num_x_frequencies)
    ray_points_batches = get_chunks(en_ray_points)
    ray_directions_norm=torch.nn.functional.normalize(ray_directions,dim=-1)
    ray_directions_norm=ray_directions_norm.unsqueeze(2).repeat(1, 1, ray_points.shape[2], 1)
    ray_directions_norm = ray_directions_norm.reshape(-1, C)
    en_ray_directions = positional_encoding(ray_directions_norm, num_d_frequencies)
    ray_directions_batches = get_chunks(en_ray_directions)
    return ray_points_batches, ray_directions_batches


"""2.4 Compute the compositing weights of samples on camera ray and then complete the volumetric rendering procedure to reconstruct a whole RGB image from the sampled points and the outputs of the neural network."""

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """

    delta=torch.zeros_like(depth_points).to(rgb.device)
    delta[...,:-1] = torch.diff(depth_points, dim=-1)
    delta[...,-1]=1e09
    sigma=nn.ReLU()
    T= torch.exp(-torch.cumsum(sigma(s).reshape(s.shape),dim=-1))
    T = T[...,None]
    temp=1-torch.exp(-sigma(s)*delta.reshape(s.shape))
    temp=temp[...,None]
    rec_image_inside=T*temp*rgb
    rec_image=torch.sum(rec_image_inside,dim=2)


    sigma_dot_delta = - F.relu(s) * delta.reshape_as(s)
    T = torch.cumprod(torch.exp(sigma_dot_delta), dim = -1)
    T = torch.roll(T,1,dims=-1)

    # Apply the volumetric rendering equation to each ray:
    C = ((T * (1 - torch.exp(sigma_dot_delta)))[..., None]) * rgb
    rec_image = torch.sum(C, dim=-2)

    return rec_image

"""2.5 Combine everything together. Given the pose position of a camera, compute the camera rays and sample the 3D points along these rays. Divide those points into batches and feed them to the neural network. Concatenate them and use them for the volumetric rendering to reconstructed the final image."""

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    #compute all the rays from the image
    R=pose[:3,:3]
    T=pose[:3,-1].reshape(-1,1)
    ray_o, ray_dir=get_rays(height,width,intrinsics,R,T)

    #sample the points from the rays
    ray_p, depth_p = stratified_samTpling(ray_o, ray_dir, near, far, samples)

    #divide data into batches to avoid memory errors
    ray_batches, ray_dir_batches = get_batches(ray_p, ray_dir, num_x_frequencies, num_d_frequencies)
    #forward pass the batches and concatenate the outputs at the enduts
    rgb_batch = []
    s_batch = []
    for ray_points, ray_directions in zip(ray_batches, ray_dir_batches):
        rgb, sigma = model.forward(ray_points, ray_directions)
        rgb_batch.append(rgb)
        s_batch.append(sigma)

    rgb = torch.cat(rgb_batch, dim=0)
    s = torch.cat(s_batch, dim=0)

    rgb = rgb.reshape(height, width, samples, -1)
    s = s.reshape(height, width, samples)

    # Apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(rgb, s, depth_p)

    return rec_image

