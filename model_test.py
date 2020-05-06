import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras import Model
from keras.layers import Input, Conv2D, Embedding, Lambda, Concatenate, BatchNormalization, \
    Dot, Add, Activation, Bidirectional, LSTM, GRU, RepeatVector, Dense, Reshape
import keras.backend as K
import matplotlib.pyplot as plt
import json
import pickle
import csv
import seaborn as sns
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model


BATCH_SIZE = 64
EPOCHS = 45
LATENT_DIM = 500
LATENT_DIM_DECODER = 250
NUM_SAMPLES = 9796
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100


def softmax_over_time(x):
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s


def stack_and_transpose(x):
    x = K.stack(x)
    x = K.permute_dimensions(x, pattern=(1, 0, 2))
    return x


def attn_loss(y_true, y_pred):
    # both are of shape N x T x D
    mask = K.cast(y_true > 0, dtype='float32')
    out = mask * y_true * K.log(y_pred)
    return -K.sum(out) / K.sum(mask)


def attn_acc(y_true, y_pred):
    # both are of shape N x T x K
    targ = K.argmax(y_true, axis=-1)
    pred = K.argmax(y_pred, axis=-1)
    correct = K.cast(K.equal(targ, pred), dtype='float32')

    # 0 is padding, don't include those
    mask = K.cast(K.greater(targ, 0), dtype='float32')
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)
    return n_correct / n_total


max_len_target = 32
num_words_output = 28

# BUILD THE MODEL -----------------------------------------------------------------------------------------------------
# ENCODER
encoder_inputs_placeholder = Input(name='EncoderInput', shape=(250, 101, 1))
x = Conv2D(32, (41, 11), strides=(2, 2), activation='relu', padding='same', name='Conv2D_1')(encoder_inputs_placeholder)
x = BatchNormalization(name='BatchNorm_1')(x)
x = Conv2D(32, (21, 11), strides=(2, 1), activation='relu', padding='same', name='Conv2D_2')(x)
x = BatchNormalization(name='BatchNorm_2')(x)
x = Conv2D(96, (21, 11), strides=(2, 1), activation='relu', padding='same', name='Conv2D_3')(x)
x = BatchNormalization(name='BatchNorm_3')(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])), name='ReshapeConv2RNN')(x)

encoder_rnn = Bidirectional(GRU(LATENT_DIM, return_sequences=True), name='EncoderRNN_bidir')
encoder_outputs = encoder_rnn(x)


decoder_inputs_placeholder = Input(shape=(max_len_target,), name='DecoderInput')
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM, name='DecoderEmbedding')
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

# ATTENTION
attn_repeat_layer = RepeatVector((conv_shape[1]), name='AttentionRepeat_S0')
attn_concat_layer = Concatenate(axis=-1, name='AttentionConcatenate')
attn_dense1 = Dense(10, activation='tanh', name='AttentionDense_1')
attn_dense2 = Dense(1, activation=softmax_over_time, name='AttentionDenseSoftmax')
attn_dot = Dot(axes=1, name='AttentionDot')


def one_step_attention(h, st_1):
    st_1 = attn_repeat_layer(st_1)
    x = attn_concat_layer([h, st_1])
    x = attn_dense1(x)
    alphas = attn_dense2(x)
    context = attn_dot([alphas, h])
    return context


# Decoder
decoder_rnn = LSTM(LATENT_DIM_DECODER, return_state=True, name='DecoderRNN')
decoder_dense = Dense(num_words_output, activation='softmax', name='DecoderFinalDense')

initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
context_last_word_concat_layer = Concatenate(axis=2, name='ContextLastWord')

s = initial_s
c = initial_c


# Main decoder loop
outputs = []
for t in range(max_len_target):
    context = one_step_attention(encoder_outputs, s)

    # we need a different layer for each time step
    selector = Lambda(lambda x: x[:, t:t + 1])
    xt = selector(decoder_inputs_x)

    # combine
    decoder_lstm_input = context_last_word_concat_layer([context, xt])

    # pass the combined [context, last word] into the LSTM
    # along with [s, c]
    # get the new [s, c] and output
    o, s, c = decoder_rnn(decoder_lstm_input, initial_state=[s, c])

    # final dense layer to get next word prediction
    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)


# Convert the list of outputs to a matrix
stacker = Lambda(stack_and_transpose, name='DecoderStacker')
outputs = stacker(outputs)

# create the model
model = Model(
    inputs=[
        encoder_inputs_placeholder,
        decoder_inputs_placeholder,
        initial_s,
        initial_c,
    ],
    outputs=outputs
)
model.compile(
    optimizer='rmsprop',
    loss=attn_loss,
    metrics=[attn_acc]
)

print(model.summary())

'''
checkpoint = ModelCheckpoint("best_model.hdf5", verbose=1,
    save_best_only=False, period=1)

# train the model
z = np.zeros((NUM_SAMPLES, LATENT_DIM_DECODER))  # initial [s, c]
r = model.fit([encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot, batch_size=BATCH_SIZE,
              epochs=EPOCHS, shuffle=True)

plt.style.use('seaborn')
plt.figure()
plt.plot(r.history['loss'])
plt.plot(r.history['accuracy'])

model.save_weights('attn_new_weights.h5')
model.save('attn_new_model.h5')

encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)
encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM*2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
context = one_step_attention(encoder_outputs_as_input, initial_s)
decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)

decoder_model = Model(
    inputs=[
        decoder_inputs_single,
        encoder_outputs_as_input,
        initial_s,
        initial_c
    ],
    outputs=[decoder_outputs, s, c]
)

encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')
plt.show()
'''