import gzip, os
import cPickle as pickle, numpy as np

def mnist_initialisation(sets):
    set_img , set_label = sets

    # set_img = set_img.reshape(set_img.shape[0],28,28,1).astype('float32')
    set_img = set_img.reshape(set_img.shape[0],784).astype('float32')
    
    set_label_vec = np.zeros((len(set_label), 10), dtype=np.float)
    for i, label in enumerate(set_label):
        set_label_vec[i,int(set_label[i])] = 1.0

    return set_img , set_label_vec


def load_mnist():
    dataset = '/home/kevinzhang/Desktop/datasets/mnist/mnist.pkl.gz'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    rval = mnist_initialisation(train_set), \
        mnist_initialisation(valid_set), \
        mnist_initialisation(test_set)

    return rval