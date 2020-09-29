from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.utils import to_categorical
import numpy as np
from myopt import Ada_heavy_ball, SGD_mom
import matplotlib.pyplot as plt
from keras import regularizers
from keras.datasets import cifar10, cifar100, mnist
from keras.optimizers import Adam, SGD, Adagrad, RMSprop
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
plt.switch_backend('agg')


def exp_sc(opt, epoch, dataset_name='cifar100'):
    num_classes = None
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
    if dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    if dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        num_classes = 100
    x_train = x_train.astype('float32') / 255
    if dataset_name == 'mnist':
        x_train = np.expand_dims(x_train, axis=-1)
    x_test = x_test.astype('float32') / 255
    if dataset_name == 'mnist':
        x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # 4-layer CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # model.summary()
    from keras.utils.training_utils import multi_gpu_model
    model = multi_gpu_model(model, gpus=2)

    model.compile(loss='categorical_crossentropy',
                       optimizer=opt,
                       metrics=['accuracy'], )

    history = model.fit(x_train, y_train, epochs=epoch, batch_size=512, validation_data=(x_test, y_test), verbose=1)
    return history.history['loss'], history.history['val_acc']


def main():
    epoch = 100
    opt_methods = [SGD_mom, Ada_heavy_ball, Adam, Adagrad, SGD, RMSprop]
    name_methods = ['SGD_momentum', 'Ada_heavy_ball', 'Adam', 'Adagrad', 'SGD', 'RMSprop']
    loss_dict = {}
    val_acc_dict = {}
    # repeat 5 times
    for i in range(5):
        for opt, name in zip(opt_methods, name_methods):
            best_loss = None
            best_val_acc = [-1]
            for lr in [0.1, 0.01, 0.001, 0.0001]:
                print("****************name={}**********lr={}*****************".format(name, lr))
                loss, val_acc = exp_sc(opt(lr=lr), epoch, dataset_name='cifar10')
                if val_acc[-1] > best_val_acc[-1]:
                    best_val_acc = val_acc
                    best_loss = loss
            loss_dict[name] = best_loss
            val_acc_dict[name] = best_val_acc
        with open('cifar10_loss_{}.pkl'.format(i+1), 'wb') as dump_file:
            pickle.dump(loss_dict, dump_file)
        with open('cifar10_val_acc_{}.pkl'.format(i+1), 'wb') as dump_file:
            pickle.dump(val_acc_dict, dump_file)


if __name__ == '__main__':
    main()
    print('hello world')
