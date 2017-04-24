import scipy.io, numpy as np

def load_svhn():

    train_mat = scipy.io.loadmat('/home/kevinzhang/Desktop/datasets/svhn/train_32x32.mat')
    test_mat = scipy.io.loadmat('/home/kevinzhang/Desktop/datasets/svhn/test_32x32.mat')

    train_set_x = np.transpose(train_mat['X'], axes=[3, 2, 0, 1]).astype('float32')
    test_set_x = np.transpose(test_mat['X'], axes=[3, 2, 0, 1]).astype('float32')

    new_train_set_x = np.zeros(shape=(train_set_x.shape[0],28,28),dtype='float32')
    new_test_set_x = np.zeros(shape=(test_set_x.shape[0],28,28),dtype='float32')

    counter = 0
    for i in train_set_x:
        new_i = (0.2126*i[0] + 0.7152*i[1] + 0.0722*i[2])/255.0
        new_train_set_x[counter] = new_i
        counter += 1

    counter = 0
    for i in test_set_x:
        new_i = (0.2126*i[0] + 0.7152*i[1] + 0.0722*i[2])/255.0
        new_test_set_x[counter] = new_i
        counter += 1

    # train_set_x = new_train_set_x.reshape(train_set_x.shape[0],1024)
    # test_set_x = new_test_set_x.reshape(test_set_x.shape[0],1024)

    train_set_x = new_train_set_x.reshape(train_set_x.shape[0],1,32,32)
    test_set_x = new_test_set_x.reshape(test_set_x.shape[0],1,32,32)

    train_set_y = train_mat['y'].flatten('F')
    test_set_y = test_mat['y'].flatten('F')

    idx10 = np.where(train_set_y == 10)
    train_set_y[idx10] = 0

    idx10 = np.where(test_set_y == 10)
    test_set_y[idx10] = 0

    rval = (train_set_x, train_set_y), (test_set_x, test_set_y)

    return rval

