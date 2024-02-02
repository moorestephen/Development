import numpy as np

np.random.seed(123)

def get_masked_indices():
    datapoints = 218
    mean = datapoints // 2
    std_dev = 50

    low = 0
    up = 217

    r = 4
    num_samples = datapoints // r

    unique = set()
    while len(unique) < num_samples:
        random_sample = np.random.normal(loc = mean, scale = std_dev)
        random_sample = int(np.clip(random_sample, low, up))
        unique.add(random_sample)

    random_samples = list(unique)
    random_samples = np.array(random_samples)
    

    return random_samples
