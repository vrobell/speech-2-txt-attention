import os
import subprocess
import wave
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from SpectrogramEstimator import SpectrogramEstimator


class DataPreparationTool:

    def __init__(self, label_file_dir, max_label_len, labels_dir='label_data',
                 format_data_dir=None, wav_data_dir='wav_data', spectre_data_dir='spectre_data'):
        self.outliers = []
        self.label_file_dir = label_file_dir
        self.max_label_len = max_label_len
        self.labels_dir = labels_dir
        self.format_data_dir = format_data_dir
        self.wav_data_dir = wav_data_dir
        self.spectre_data_dir = spectre_data_dir

    def process_data(self, fft_size, step_size, threshold, low_cut, high_cut, pad_len):
        ''' Processes the data and the labels to padded spectrograms and padded sequences
            for the network training '''

        if self.format_data_dir:
            print('CONVERTING THE DATA TO .WAV FORMAT...')
            self.convert2wav(self.format_data_dir, self.wav_data_dir, input_format='Mp3')

        print('ANALYZING DATA LENGTH...')
        self._detect_outliers(limit_len=pad_len)
        print('COMPUTING LOG SPECTROGRAMS...')
        self._compute_spectre_data(fft_size=fft_size,
                                   step_size=step_size,
                                   threshold=threshold,
                                   low_cut=low_cut,
                                   high_cut=high_cut,
                                   pad_len=pad_len
                                   )

        print('CONVERTING LABELS...')
        self._handle_labels_coommon_voice()
        self._rename_spectre_files()

    def _compute_spectre_data(self, fft_size, step_size, threshold, low_cut, high_cut, pad_len):
        ''' Computes the log spectrogram of every file from wav_data_dir
            and saves it as a 3D numpy array in spectre_data_dir '''

        spectre_estimator = SpectrogramEstimator(fft_size=fft_size,
                                                 step_size=step_size,
                                                 threshold=threshold,
                                                 low_cut=low_cut,
                                                 high_cut=high_cut
                                                 )

        # process the data from the wav directory
        for file in os.listdir(self.wav_data_dir):
            if file.title().lower() not in self.outliers:

                sample = self.read_wav(self.wav_data_dir + '/' + file.title().lower())
                fs = sample['fs']
                data = sample['data']
                data_spectrogram = spectre_estimator.compute_spectrogram(data=data,
                                                                         fs=fs,
                                                                         const_pad_len=pad_len
                                                                         )
                # cut the spectrogram as 4kHz
                data_spectrogram = data_spectrogram[:, :101]
                # normalize the data between 0 and 1
                data_spectrogram = (data_spectrogram - np.min(data_spectrogram)) / \
                                   (np.max(data_spectrogram)-np.min(data_spectrogram))
                # add the 3rd dimention for the network
                data_spectrogram = np.atleast_3d(data_spectrogram)
                data_spectrogram = data_spectrogram.reshape(-1, int(pad_len*100), 101, 1)
                # save the result to a file
                np.save(self.spectre_data_dir+'/'+file.title().lower().replace('.wav', ''), data_spectrogram)

    @staticmethod
    def read_wav(file_dir):
        ''' Reads in the wav file as a dict:
            {data, sampling_frequency} '''

        with wave.open(file_dir, 'rb') as smp:
            data = np.fromstring(smp.readframes(-1), 'Int16')
            fs = smp.getframerate()
            sample = {'data': data, 'fs': fs}
        return sample

    @staticmethod
    def convert2wav(input_dir, output_dir, input_format):
        ''' Converts the audio files (Mp3/Flac/Wav) to the wav format
            from the input directory and subdirectories
            to the output directory '''

        for sub_dir in os.walk(input_dir+'/'):
            curr_dir = sub_dir[0].replace('\\', '/')

            for file in os.listdir(curr_dir + '/'):
                if '.txt' not in file.title().lower():
                    subprocess.call(['ffmpeg', '-i', curr_dir + '/' + file.title().lower(),
                                     output_dir + '/' + file.title().replace(input_format, 'wav')])

    def _detect_outliers(self, limit_len):
        ''' Detects wav files longer than the limit_len[s]
            and saves them to self.outliers array '''

        for file in os.listdir(self.wav_data_dir):
            smp = self.read_wav(self.wav_data_dir + '/' + file.title().lower())
            smp_len = len(smp['data'])
            smp_len = smp_len / smp['fs']

            if smp_len > limit_len:
                self.outliers.append(file.title().lower())

    def _handle_labels_coommon_voice(self):
        ''' Processes the label data from self.label_file_dir
            to self.labels_dir as a separate file per label '''

        label_dir = self.label_file_dir.split('/')[-1]
        label_dir = label_dir.split('.')[0]

        y_data = pd.read_csv(self.label_file_dir)
        y_data = y_data.drop(['up_votes', 'down_votes', 'age', 'gender', 'accent', 'duration'], axis=1)
        y_data['filename'] = y_data['filename'].str.replace(label_dir+'/', '', regex=True)
        y_data['filename'] = y_data['filename'].str.replace('mp3', 'wav', regex=True)
        y_data['text'] = y_data['text'].str.replace("'", '', regex=True)

        # remove the wav outliers from labels
        for elem in self.outliers:
            y_data = y_data[y_data['filename'] != elem]

        y_filenames = y_data['filename']
        y_filenames = y_filenames.str.replace('wav', 'npy', regex=True)
        y_filenames = y_filenames.values

        y_data = y_data.drop('filename', axis=1)
        y_data = y_data.values[:, 0]

        # Split the label data for Teacher forcing
        target_txt_out = y_data + '$'
        target_txt_in = '&' + y_data

        # Tokenize labels
        # tokenizer = Tokenizer(char_level=True)
        # tokenizer.fit_on_texts(target_txt_in + '$')

        # with open('Tokenizer.pickle', 'wb') as handle:
        #    pickle.dump(tokenizer, handle)

        with open('Tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        target_seq_in = tokenizer.texts_to_sequences(target_txt_in)
        target_seq_out = tokenizer.texts_to_sequences(target_txt_out)
        target_seq_in, target_seq_out, y_filenames = self._remove_long_labeled(target_seq_in, target_seq_out, y_filenames)

        target_seq_in = pad_sequences(target_seq_in, maxlen=self.max_label_len, padding='post', truncating='post')
        target_seq_out = pad_sequences(target_seq_out, maxlen=self.max_label_len, padding='post', truncating='post')

        np.save(self.labels_dir+'/'+'target_seq_in', target_seq_in)
        np.save(self.labels_dir+'/'+'target_seq_out', target_seq_out)

    def _remove_long_labeled(self, y_seq_in, y_seq_out, y_filenames):
        ''' Removes the files and labels
        that labels are longer than self.max_label_len '''

        count = 0
        for i, seq in enumerate(y_seq_in):
            if len(seq) > self.max_label_len:
                y_seq_in = np.delete(y_seq_in, i-count, axis=0)
                y_seq_out = np.delete(y_seq_out, i-count, axis=0)

                if os.path.exists(self.spectre_data_dir+'/'+y_filenames[i-count]):
                    os.remove(self.spectre_data_dir+'/'+y_filenames[i-count])

                y_filenames = np.delete(y_filenames, i-count, axis=0)
                count += 1
        return y_seq_in, y_seq_out, y_filenames

    def _rename_spectre_files(self):
        ''' Renames the spectre files to their indexes '''

        i = 0
        for file in os.listdir(self.spectre_data_dir):
            os.rename(self.spectre_data_dir+'/'+file.title(), self.spectre_data_dir+'/'+str(i)+'.npy')
            i += 1


dpt = DataPreparationTool(
    label_file_dir='cv-valid-train.csv',
    max_label_len=60,
    labels_dir='labels_testing',  # 'data/train/5s/labels',
    format_data_dir='data/train/common-voice',
    wav_data_dir='dataset_wav/train',  # 'data/train/5s/wav',
    spectre_data_dir='spectre_testing'  # 'data/train/5s/spectre'
)

dpt.process_data(
    fft_size=960,
    step_size=480,
    threshold=3.0,
    low_cut=500,
    high_cut=15000,
    pad_len=2.5
)


