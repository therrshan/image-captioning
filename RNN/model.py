from keras.models import Model
from keras.initializers import RandomUniform
from keras.layers import Input, Embedding, GRU, Dense, Dropout, add

def define_model(vocab_size, max_length, EMBEDDING_DIM=300): 

    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.125)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            trainable=True,
            mask_zero=True,
            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05)
        )(inputs2) 
    se2 = Dropout(0.125)(se1) 
    se3 = GRU(256)(se2)

    decoder1 = add([fe2, se3]) 
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    return model

