from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    H, W, C = image.shape
    total_pixels = H * W
    
    pixel_indices = np.random.choice(total_pixels, size=num_clusters, replace=False)
    
    centroids_init = np.zeros((num_clusters, C))
    for i, idx in enumerate(pixel_indices):
        row = idx // W
        col = idx % W
        centroids_init[i] = image[row, col]
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # Restructure image to work easier with it
    h, w, c = image.shape
    
    assignments = np.random.randint(0, centroids.shape[0], (h,w,1))
    new_centroids = centroids.copy()
    # *** START YOUR CODE ***
    for i in range(0, max_iter):
        assignments_old = assignments.copy()
        if i % print_every == 0:
            print(f"Iteration {i}/{max_iter}: K-means clustering in progress...")
            if i > 0:
                centroid_change = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
                print(f"  Average centroid movement: {centroid_change:.4f}")
        
        # Update cluster assignments
        assignments = calculate_closest_centroids(new_centroids, image)[:,:,np.newaxis]
        
        # Update cluster means
        for j in range(0, new_centroids.shape[0]):
            # Find all pixels assigned to this centroid
            mask = (assignments[:, :, 0] == j)
            
            if np.any(mask):
                # Get pixel values for this cluster
                cluster_pixels = image[mask]
                # Calculate mean color for this centroid
                new_centroids[j] = np.mean(cluster_pixels, axis=0)
        
        # Converge and break out if assignments are the same:
        if np.array_equal(assignments, assignments_old):
            break
    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    closest_clusters = calculate_closest_centroids(centroids=centroids, image=image)
    
    # Reconstruct image
    new_image = centroids[closest_clusters]

    return new_image.astype(np.uint8)

def calculate_closest_centroids(centroids, image):
        # Flatten the first two dimensions of image to get (H*W, C) array
    h, w, c = image.shape
    flattened_image = image.reshape(h * w, c)
    
    # Calculate for each pixel the distance from all centroids in a (h * w, num_centroids, 1) array
    differences = np.linalg.norm(centroids[np.newaxis, :] - flattened_image[:, np.newaxis, :]
, axis=2)
    
    # Closest assignments per pixel
    return np.argmin(differences, axis=1).reshape(h, w)
    
def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = DATA_DIR / 'orig_small.png'
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = DATA_DIR / 'orig_large.png'
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = DATA_DIR / 'updated_large.png'
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default=DATA_DIR / 'peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default=DATA_DIR / 'peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
