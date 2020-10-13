import numpy as np
import neuralnetwork as nn
from Autoencoder import autoencoder
import os
import struct
import pickle

mnist_path = "http://yann.lecun.com/exdb/mnist/"

zip_list = ["train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz" ]

ext_list = ["train-images.idx3-ubyte",
            "train-labels.idx1-ubyte",
            "t10k-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte" ]

def downloadMNIST(path = ".", unzip=True):
    import urllib.request
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(len(zip_list)):
        zip_name = zip_list[i]
        fname1 = os.path.join(path, zip_name)
        print("Download ", zip_name, "...")
        if not os.path.exists(fname1):
            urllib.request.urlretrieve( mnist_path + zip_name, fname1)
        else:
            print("pass")

        if unzip:
            import gzip
            import shutil
            ext_name = ext_list[i]
            fname2 = os.path.join(path, ext_name)
            print("Extract", fname1, "...")
            if not os.path.exists(fname2):
                with gzip.open(fname1, 'rb') as f_in:
                    with open(fname2, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                print("pass")

def loadMNIST(dataset = "training", path = "."):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)
    fimg.close()

    return img, lbl



if __name__ == '__main__':
    #downloadMNIST(path='MNIST_data', unzip=True)

    print("Training...")
    num_classes = 10
    x_train, y_train = loadMNIST(dataset="training", path="MNIST_data")
    x_validation = x_train[55000:]
    y_validation = y_train[55000:]
    x_test, y_test = loadMNIST(dataset="testing", path="MNIST_data")
    test_targets = np.array([y_test]).reshape(-1)
    test_one_hot_targets = np.eye(num_classes)[y_test]

    targets = np.array([y_train]).reshape(-1)
    one_hot_targets = np.eye(num_classes)[y_train]

    ''' Wide Model '''
    # train
    net = nn.neural_network(3, [784, 256, 10], [None, "relu", "softmax"], cost_function="cross_entropy")
    net.train(100, inputs=x_train[0:50000], labels=one_hot_targets[0:50000],
            validation_input=x_validation,validation_targets=y_validation, num_epochs=100, learning_rate=0.000001)

    print("Testing...")
    # test

    test_predict_result = net.predict(x_test[0:100])
    print(" --------- result -----------")
    print(test_predict_result)
    predict_labels=[]
    for arr in test_predict_result:
        label = np.argmax(arr)
        predict_labels.append(label)
    net.caculate_accuracy(predict_labels, y_test[0:100])


    # ''' Deep Model '''
    # deep_network = nn.neural_network(4, [784, 204, 202, 10], [None, "relu", "relu", "softmax"], cost_function="cross_entropy")
    # deep_network.train(100, inputs=x_train[0:55000], labels=one_hot_targets[0:55000],
    #         validation_input=x_validation,validation_targets=y_validation, num_epochs=100, learning_rate=0.00005)
    #
    # print("Testing...")
    #
    # test_predict_result = deep_network.predict(x_test[0:100])
    #
    # print("Deep Network Result")
    # predict_labels=[]
    # for arr in test_predict_result:
    #     label = np.argmax(arr)
    #     predict_labels.append(label)
    # deep_network.caculate_accuracy(predict_labels, y_test[0:100])


    # '''Autoencoder'''
    #
    # autoencoder_target = np.divide(x_train, 255)
    # print(autoencoder_target[0])
    #
    # Autoencoder = autoencoder(3, [784, 128, 784], [None, "relu", "sigmoid"], cost_function="mean_squared")
    # Autoencoder.train(600, inputs=x_train, labels=autoencoder_target,
    #         validation_input=x_validation,validation_targets=y_validation, num_epochs=100, learning_rate=0.0001)
    # Autoencoder.get_encoder(x_test[0:600])
    # with open('label_list.pickle', 'wb') as wf:
    #     pickle.dump(y_test[0:600], wf)
    # with open('testing_data.pickle', 'wb') as test_wf:
    #     pickle.dump(x_test[0:600], test_wf)
    #
    # print("Testing...")
    # # test
    #
    # test_predict_result = net.predict(x_test[0:100])
