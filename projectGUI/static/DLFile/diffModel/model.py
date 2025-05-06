# model.py
from keras import layers, models
from keras.regularizers import l2


def se_attention(input_feature, reduction_ratio=8):
    channel = input_feature.shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(input_feature)
    excitation = layers.Dense(channel // reduction_ratio, activation='relu')(squeeze)
    excitation = layers.Dense(channel, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, channel))(excitation)
    return layers.Multiply()([input_feature, excitation])


def create_sunset_model(input_shape=(64, 64, 3)):
    num_filters = 24
    kernel_size = (3, 3)
    pool_size = (2, 2)
    dense_size = 1024
    drop_rate = 0.4

    x_in = layers.Input(shape=input_shape)

    x = layers.Conv2D(12, kernel_size, padding="same", activation='relu')(x_in)
    x = layers.BatchNormalization()(x)
    x = se_attention(x)
    x = layers.MaxPooling2D(pool_size)(x)

    x = layers.Conv2D(24, kernel_size, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = se_attention(x)
    x = layers.MaxPooling2D(pool_size)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_size, activation='relu')(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(dense_size, activation='relu')(x)
    x = layers.Dropout(drop_rate)(x)

    y_out = layers.Dense(1)(x)

    model = models.Model(inputs=x_in, outputs=y_out)
    return model