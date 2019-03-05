'''
Heirarchical division, until K-clusters are obtained
http://infolab.stanford.edu/~ullman/mmds/ch10.pdf
http://ai.stanford.edu/~ang/papers/nips01-spectral.pdf
http://www.sfu.ca/personal/archives/richards/Pages/london98.pdf
'''


#from _future_ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
import pandas as pd
import numpy as np

batch_size=128
epochs = 10
output = 10

def CNN(dataset,test_percentage=0.2):
    bound = int(round(len(dataset)*test_percentage))
    test,train = dataset.loc[:dataset.shape[0]-(bound+1)],dataset.loc[dataset.shape[0]-bound:]
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test, y_test  = np.array(test.loc[:,1:])  , test[0]
    x_train,y_train = np.array(train.loc[:,1:]) ,train[0]

    x_train = x_train.reshape(x_train.shape[0],28,28,1)
    x_test = x_test.reshape(x_test.shape[0],28,28,1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()

    model.add(Conv2D(96,kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),strides=(1,1),padding='valid'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='valid'))

    model.add(BatchNormalization(momentum=0.99,\
                                 epsilon=0.001, center=True, scale=True, beta_initializer='zeros',\
                                 gamma_initializer='ones', moving_mean_initializer='zeros', \
                                 moving_variance_initializer='ones', beta_regularizer=None, \
                                 gamma_regularizer=None, beta_constraint=None,\
                                 gamma_constraint=None))

    model.add(Conv2D(256, kernel_size=(11,11),activation='relu',strides=1,padding='valid'))

    model.add(MaxPooling2D(pool_size=(3,3),strides=2,padding='valid'))

    model.add(BatchNormalization(momentum=0.99,\
                                 epsilon=0.001, center=True, scale=True, beta_initializer='zeros',\
                                 gamma_initializer='ones', moving_mean_initializer='zeros', \
                                 moving_variance_initializer='ones', beta_regularizer=None, \
                                 gamma_regularizer=None, beta_constraint=None,\
                                 gamma_constraint=None))

    model.add(Conv2D(384, kernel_size=(3,3),activation='relu',strides=1,padding='valid'))

    model.add(Conv2D(384, kernel_size=(3, 3),activation='relu',strides=1,padding='valid'))
    model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',strides=1,padding='valid'))


    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(4096,activation='relu'))

    model.add(Dropout(0.5))


    model.add(Dense(4096,activation='relu'))

    model.add(Dense(10,activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]*100)


df = pd.read_csv('/home/clabuser/ACHINT/LA-IP_OP/input_second_problem/mnist_train.csv',index_col=False,header=None)

CNN(df)