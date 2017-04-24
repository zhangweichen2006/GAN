import gzip, os
import cPickle as pickle, numpy as np
from scipy.misc import imresize

def usps_initialisation(sets,add_pad=False):
    set_img , set_label = sets

    counter = 0
    new_img_set = np.zeros(shape=(set_img.shape[0],28,28),dtype='float32')

    if add_pad == False:
        set_img = set_img.reshape(set_img.shape[0],256).astype('float32')
        for n in set_img:
            n = np.reshape(n,(16,16))
            new_img = imresize(n, (28, 28))
            new_img_set[counter] = new_img
            counter += 1
    else:
        
        for x in set_img:
            zero = np.float32(0)
            x_temp = np.insert(x,0,[zero for i in xrange(28*6)])
            for n in xrange(16):
                x_temp = np.insert(x_temp,(n+6)*28,[zero for i in xrange(6)])
                x_temp = np.insert(x_temp,(n+6)*28+22,[zero for i in xrange(6)])
            x_temp = np.insert(x_temp,616,[zero for i in xrange(28*6)])
            new_img = np.reshape(x_temp,(28,28))
            new_img_set[counter] = new_img
            counter += 1                               

    set_img = (new_img_set.reshape(new_img_set.shape[0], 28, 28, 1).astype('float32'))/255.

    set_label_vec = np.zeros((len(set_label), 10), dtype=np.float)
    for i, label in enumerate(set_label):
        set_label_vec[i,int(set_label[i])] = 1.0

    return set_img , set_label_vec


def load_usps(add_pad=False):
    dataset = '/home/kevinzhang/Desktop/datasets/usps/usps.pkl.gz'

    f = gzip.open(dataset, 'rb')
    train_set, test_set = pickle.load(f)
    f.close()

    rval = usps_initialisation(train_set,add_pad), \
        usps_initialisation(test_set,add_pad)

    return rval