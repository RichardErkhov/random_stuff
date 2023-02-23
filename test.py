import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, BatchNormalization, CuDNNLSTM,Activation, Reshape
from keras import layers
from keras import losses
from keras.optimizers import Adam,SGD, RMSprop, Nadam
from keras.regularizers import l1, l2
import keras.models as models
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input):
    ''' Create a spatial squeeze-excite block

    Args:
        input: input tensor

    Returns: a keras tensor

    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input)

    x = multiply([input, se])
    return x


def channel_spatial_squeeze_excite(input, ratio=16):
    ''' Create a spatial squeeze-excite block

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x

def model_creator_test3(leng=448, layer_size=512, depth=8):
  #linear network
  model = Sequential()

  #model.add(Embedding(3+1, 8, input_length=leng))
  #model.add(Flatten())

  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu", input_dim=leng))

  model.add(BatchNormalization())
  #model.add(Dropout(0.2))  
  for i in range(depth):
    model.add(Dense(units=layer_size, kernel_initializer='normal'))#, activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
  model.add(Dense(units=1, activation="sigmoid"))

  #opt = SGD(lr=0.0001, momentum=0.9, nesterov=True) 
  opt = Adam(lr=0.001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model

def model_creator_test10():
  inputx = layers.Input(shape=(448,))
  conv_depth= 10
  conv_size= 128
  se_ratio = 4
  x = layers.Reshape((7,8,8))(inputx)
  x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
  for _ in range(conv_depth):
    previous = x
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
    x = squeeze_excite_block(x, se_ratio)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
    x = squeeze_excite_block(x, se_ratio)
    #x = layers.BatchNormalization()(x)
    x = layers.Add()([x, previous])
    x = layers.Activation('relu')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(units=1, activation="sigmoid")(x)
  model = models.Model(inputs=inputx, outputs=x)
  #model.add(Dense(units=1, activation="sigmoid"))
  opt = Adam(lr=0.001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  return model
def model_creator():
  inputx = layers.Input(shape=(448,))
  conv_depth= 6
  conv_size= 64
  se_ratio = 4
  x = layers.Reshape((7,8,8))(inputx)
  x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
  for _ in range(conv_depth):
    previous = x
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
    x = squeeze_excite_block(x, se_ratio)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
    x = squeeze_excite_block(x, se_ratio)
    #x = layers.BatchNormalization()(x)
    x = layers.Add()([x, previous])
    x = layers.Activation('relu')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(units=1, activation="sigmoid")(x)
  model = models.Model(inputs=inputx, outputs=x)
  #model.add(Dense(units=1, activation="sigmoid"))
  opt = Adam(lr=0.001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  return model

def model_creator_test7():
  inputx = layers.Input(shape=(448,))
  conv_depth= 8
  conv_size= 64
  x = layers.Reshape((7,8,8))(inputx)
  x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
  for _ in range(conv_depth):
    previous = x
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, previous])
    x = layers.Activation('relu')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(units=1, activation="sigmoid")(x)
  model = models.Model(inputs=inputx, outputs=x)
  #model.add(Dense(units=1, activation="sigmoid"))
  opt = Adam(lr=0.001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  return model

def model_creator_xx(leng=448, layer_size=512):
  model = Sequential()
  model.add(Reshape((7, 8, 8)))
  model.add(Flatten())
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  model.add(Dense(units=1, activation="sigmoid"))
  opt = Adam(lr=0.001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  return model

def model_creator_test5(leng=448, layer_size=2048):
  
  #linear network
  model = Sequential()

  #model.add(Embedding(3+1, 8, input_length=leng))
  #model.add(Flatten())

  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu", input_dim=leng))
  model.add(BatchNormalization())
  for i in range(4):
    model.add(Dense(units=layer_size//(2**i), kernel_initializer='normal'))#, activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
  #for i in range(4):
    #for j in range(16):
      #model.add(Dense(units=layer_size//(2**i), kernel_initializer='normal'))#, activation="relu"))
      #model.add(BatchNormalization())
      #model.add(Activation('relu'))
      #model.add(Dropout(0.01))
  #model.add(Dropout(0.2))  
  #model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  #model.add(Dropout(0.2))
  #model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  #model.add(Dropout(0.2))
  #model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  model.add(Dense(units=1, activation="sigmoid"))

 # opt = SGD(lr=0.0001, momentum=0.9, nesterov=True) 
  #opt = Adam(lr=0.001)
  #opt = SGD(lr=0.002)
  opt = Nadam(lr=0.0005)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model

#1538
def model_creator_x(leng=448, layer_size=512):

  #linear network
  model = Sequential()

  #model.add(Embedding(3+1, 8, input_length=leng))
  #model.add(Flatten())

  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu", input_dim=leng))

  model.add(BatchNormalization())
  #model.add(Dropout(0.2))  
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  #model.add(Dropout(0.2))
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  #model.add(Dropout(0.2))
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  model.add(Dense(units=1, activation="sigmoid"))

  #opt = SGD(lr=0.0001, momentum=0.9, nesterov=True) 
  opt = Adam(lr=0.001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model
