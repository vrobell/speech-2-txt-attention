import numpy as np
from keras import Model
from keras.layers import Input, Conv2D, Embedding, Lambda, Concatenate, BatchNormalization, \
    Dot, Bidirectional, LSTM, GRU, RepeatVector, Dense, Reshape
import keras.backend as K
import matplotlib.pyplot as plt
from AttnDataGenerator import AttnDataGenerator
from keras.utils import to_categorical


# SET PARAMS ---------------------------------------------------------
# Directories
weights_file_dir = 'files/attn_final_weights_2s.h5'  # File with previous weights
X_data_dir = 'spectre_testing'   # Folder with spectograms
on_epoch_dir = 'on_epoch'        # Folder to store the weights after every epoch
models_dir = 'models'            # Folder to store the final model
y_labels_dir = 'labels_testing'  # Folder with labels

# Training params
BATCH_SIZE = 128
EPOCHS = 15
value_split = 133500 - BATCH_SIZE  # Number of data files for training

# Some model params
spectre_seq_dim = 500
max_len_target = 120
num_words_output = 29
LATENT_DIM = 500
LATENT_DIM_DECODER = 250
EMBEDDING_DIM = 100

# ---------------------------------------------------------------------
y_data_in = np.load(y_labels_dir + '/target_seq_in.npy')
y_data_out = np.load(y_labels_dir + '/target_seq_out.npy')
y_data_out = to_categorical(y_data_out)

gen = AttnDataGenerator(
    Y_data_in=y_data_in,
    Y_data_out=y_data_out,
    X_data_location=X_data_dir,
    X_data_width=spectre_seq_dim,
    batch_size=BATCH_SIZE,
    word_idx_len=num_words_output,
    on_epoch_dir=on_epoch_dir,
    val_split=value_split,
    latent_dim_decoder=LATENT_DIM_DECODER
)


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


# BUILD THE MODEL -------------------------------------------------------------
# ENCODER
encoder_inputs_placeholder = Input(name='EncoderInput', shape=(spectre_seq_dim, 101, 1))
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


# COMPILE THE MODEL --------------------------------------------------------------------
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
model.load_weights(weights_file_dir)


# TRAIN THE MODEL ------------------------------------------------------------------------
train_hist = model.fit_generator(
    generator=gen.next_train(),
    epochs=EPOCHS,
    steps_per_epoch=value_split // BATCH_SIZE,
    initial_epoch=0,
    callbacks=[gen]
)


# VISUALIZE RESULTS ----------------------------------------------------------------------
plt.figure()
plt.title('Training loss')
plt.style.use('seaborn')
plt.plot(train_hist.history['loss'])

plt.figure()
plt.title('Training accuracy')
plt.plot(train_hist.history['attn_acc'])
plt.show()


# SAVE FINAL MODEL -----------------------------------------------------------------------
model.save_weights(models_dir + '/attn_final_weights_5s.h5')
model.save(models_dir + '/attn_whole_model_5s.h5')

encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)
encoder_outputs_as_input = Input(shape=(conv_shape[1], LATENT_DIM*2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
context = one_step_attention(encoder_outputs_as_input, initial_s)
decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])
o, s, c = decoder_rnn(decoder_lstm_input, initial_state=[initial_s, initial_c])
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

encoder_model.save(models_dir + '/encoder_model_5s.h5')
decoder_model.save(models_dir + '/decoder_model_5s.h5')
