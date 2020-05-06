import numpy as np
from scipy.signal import lfilter, butter


class SpectrogramEstimator:

    def __init__(self, fft_size, step_size, threshold=3, low_cut=500, high_cut=15000):
        self.fft_size = fft_size
        self.step_size = step_size
        self.threshold = threshold
        self.low_cut = low_cut
        self.high_cut = high_cut

    def compute_spectrogram(self, data, fs, const_pad_len):
        ''' Pads the signal to the const. length;
            Filters the signal;
            Computes the log spectrogram of the signal '''

        data = data*(1/max(data))
        padding = np.zeros(int(const_pad_len*fs - len(data)))
        data = np.concatenate((data, padding))
        data = self._bandpass_filter(data, order=1, low_cut=self.low_cut, high_cut=self.high_cut, fs=fs)

        log_spectrogram = self._log_spectrogram(
            data.astype('float64'),
            fft_size=self.fft_size,
            step_size=self.step_size,
            threshold=self.threshold
        )

        return log_spectrogram

    @staticmethod
    def _bandpass_filter(data, order, low_cut, high_cut, fs):
        ''' Filters the signal with the butter bandpass filter '''

        fn = 0.5*fs
        low_cut = low_cut/fn
        high_cut = high_cut/fn
        b, a = butter(order, [low_cut, high_cut], btype='band')
        data_filtered = lfilter(b, a, data)
        return data_filtered

    @staticmethod
    def _window_overlap(sample, win_size, step_size):
        ''' Overlaps the sliding window '''

        # for all windows to fit in the signal len
        fill = np.zeros((win_size - len(sample) % win_size))
        sample = np.hstack((sample, fill))

        valid_range = len(sample) - win_size
        nw = valid_range // step_size
        overlapped = np.ndarray((nw, win_size), dtype=sample.dtype)

        for i in np.arange(nw):
            low_bnd = i * step_size
            high_bnd = low_bnd + win_size
            overlapped[i] = sample[low_bnd:high_bnd]

        return overlapped

    def _stft(self, sample, fft_size, step_size):
        ''' Computes the STFT of a 1D signal '''

        sample -= sample.mean()
        local_fft = np.fft.fft
        cut = fft_size//2

        sample = self._window_overlap(sample, fft_size, step_size)
        win = 0.54 - 0.46*np.cos(2*np.pi*np.arange(fft_size) / (fft_size-1))
        sample = sample*win[None]
        sample = local_fft(sample)[:, :cut]
        return sample

    def _log_spectrogram(self, sample, fft_size, step_size, threshold):
        ''' Computes the log spectrogram of a 1D signal '''

        spectrogram = self._stft(sample, fft_size, step_size)
        spectrogram = np.abs(spectrogram)
        spectrogram /= spectrogram.max()
        spectrogram = np.log10(spectrogram)
        spectrogram[spectrogram < -threshold] = -threshold

        return spectrogram
