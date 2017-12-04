import keras
from keras.layers import Dense, Input
from keras.models import Model

def dense_network(
    input_shape,
    net_structure=[
    (50, 'Dense_1'),
    (20, 'Dense_2'),
    ( 5, 'Dense_3')],
    g='relu',
    optim='rmsprop',
    l=keras.losses.categorical_crossentropy):

    a = Input(shape=input_shape)
    x = a
    for n_units, name in net_structure:
        x = Dense(n_units,activation=g,name=name)(x)

    x = Dense(2,activation='softmax',name='readout')(x)

    model = Model(inputs=a,outputs=x)
    
    model.compile(optimizer=optim,
              loss=l,
              metrics=['accuracy'])
    
    return model