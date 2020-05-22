import numpy as np
import keras


class AttnDataGenerator(keras.callbacks.Callback):
    def __init__(
            self, Y_data_in, Y_data_out, X_data_location, X_data_width,
            batch_size, word_idx_len, on_epoch_dir, val_split, latent_dim_decoder
    ):
        self.batch_size = batch_size
        self.Y_data_in = Y_data_in
        self.Y_data_out = Y_data_out
        self.X_loc = X_data_location
        self.X_width = X_data_width
        self.word_idx_len = word_idx_len
        self.val_split = val_split
        self.on_epoch_dir = on_epoch_dir
        self.curr_train_index = 0
        self.latent_dim_decoder = latent_dim_decoder


    @staticmethod
    def replace_val(map_dict, array):
        new_arr = np.copy(array)
        for k, x in map_dict.items():
            new_arr[array == k] = x

        return new_arr

    def __len__(self):
        return int(np.floor(self.Y_data_in.shape[0] / self.batch_size))

    def get_output_size(self):
        return self.word_idx_len

    def get_batch(self, index, size):
        X_data = np.empty((size, self.X_width, 101, 1))
        for i in range(index, index+size):
            X_data[i-index, :, :, :] = np.load(self.X_loc+'/'+str(i)+'.npy')

        y_in = self.Y_data_in[index:index+size, :]
        y_out = self.Y_data_out[index:index+size, :]
        z = np.zeros((size, self.latent_dim_decoder))

        inputs = [X_data, y_in, z, z]
        outputs = y_out

        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.curr_train_index, self.batch_size)
            self.curr_train_index += self.batch_size

            if self.curr_train_index >= self.val_split:
                self.curr_train_index = self.curr_train_index % self.batch_size
            yield ret

    def on_train_begin(self, logs={}):
        self.curr_train_index = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.curr_train_index = 0

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(self.on_epoch_dir+'/weights%02d.h5' % (epoch))