# coding: utf-8
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os

batch_size = 128
num_classes = 10
epochs = 12
data_augmentation = True
num_predictions = 20
# 路径拼接，可传入多个路径
save_dir = os.path.join(os.getcwd(), 'save_models')
model_name = 'keras_cifar10_trained_model.h5'


# 获取训练集，测试集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(63, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.00011, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /=255
x_test /=255

if data_augmentation:
    print('no data augmentation')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        shuffle=True)

else:
    print('using real-time data augmentation')
    # 图像数据集扩充
    dataGen = ImageDataGenerator(
        # 数据集去中心化
        featurewise_center=False,
        # 样本均值为0
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        # 对数据加以白化ZCA
        zca_whitening=False,
        zca_epsilon=1e-06,
        # 数据提升时，随机转的角度
        rotation_range=0,
        # 图片宽度比例
        width_shift_range=0.1,
        height_shift_range=0.1,
        # 剪切强度
        shear_range=0,
        # 随机缩放
        zoom_range=0,
        # 随机隧道偏移幅度
        channel_shift_range=0.,
        fill_mode='nearest',
        # 边界填充值
        cval=0,
        # 水平反转
        horizontal_flip=True,
        # 竖直翻转
        vertical_flip=True,
        # 将0-255值映射到0-1之间
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)
    dataGen.fit(x_train)

    model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('saved trained modelat %s' % model_path)
scores = model.evaluate(x_test, y_test, verbose=1)
print('test loss : ', scores[0])
print('test accuracy', scores[1])
