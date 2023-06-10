import numpy as np

def compute_mean_vector(volume):
    # Input volume should have dims (C, T, H, W)
    mean_vector = np.mean(volume, axis=(1,2,3))
    return mean_vector

def compute_std_vector(volume):
    # Input volume should have dims (C, T, H, W)
    std_vector = np.std(volume, axis=(1, 2, 3))
    return std_vector

##### Code for testing purposes #####
if __name__ == '__main__':
    # Example input volume of shape (C, T, H, W)
    input_volume = np.random.rand(3, 10, 256, 256)

    # Compute the mean vector
    mean_vector = compute_mean_vector(input_volume)

    print("Mean vector shape:", mean_vector.shape)
    print("Mean vector:", mean_vector)

    # Compute the standard deviation vector
    std_vector = compute_std_vector(input_volume)

    print("Standard deviation vector shape:", std_vector.shape)
    print("Standard deviation vector:", std_vector)