from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential


def create_cnn(height, width, depth, filters=(16, 32, 64)):
    inputShape= (height, width, depth)
    model= Sequential()

    for filter in filters:
        if filter == 16:
            model.add(Conv2D(filter, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=inputShape))
        else:
            model.add(Conv2D(filter, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

    model.add(Flatten())

    return model

