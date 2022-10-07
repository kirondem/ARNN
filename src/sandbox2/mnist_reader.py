import os


PATH  =  os.path.dirname(os.path.abspath(__file__))

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    path = os.path.join(PATH, '..', 'data', 'fashion-mnist','images-idx3-ubyte.npy')
    with open(path, 'wb') as f:
        np.save(f, images)

    path = os.path.join(PATH, '..', 'data', 'fashion-mnist','labels-idx1-ubyte.npy')
    with open(path, 'wb') as f:
        np.save(f, labels)
    
    return images, labels

if __name__ == "__main__":
    images, labels = load_mnist(os.path.join(PATH, '..', 'data', 'fashion-mnist'), kind='train')
    print(images.shape)