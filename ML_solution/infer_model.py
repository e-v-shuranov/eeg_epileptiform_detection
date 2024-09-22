
import numpy as np
from sympy.polys.matrices.sdm import sdm_nullspace_from_rref

from utils_pipeline import Pipeline
from preprocessing_library import FFT, Slice, Magnitude, Log10
import re
import pyedflib

import pandas as pd
import os
# import torch
# import csv
# import torch.nn as nn



# import simple_EEGNet from simple_baseline
from simple_baseline import *
from model_cnn_lstm_baseline import *
# from test_models import *
# from simple_baseline import simple_EEGNet

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_signal(f, signal_labels, electrode_name, start, stop):
    """
    f - opened edf file.
    signal_labels - list of signals.
    electrode_name - the name of electrode.
    start - start of the window in seconds.
    stop - end of the window in seconds.
    """
    tuh_label = [s for s in signal_labels if 'EEG ' + electrode_name + '-' in s]
    if len(tuh_label) > 1:
        print(tuh_label)
        exit('Multiple electrodes found with the same string! Abort')

    channel = signal_labels.index(tuh_label[0])
    signal = np.array(f.readSignal(channel))
    if start > len(signal) or stop > len(signal):
        return [signal, True]
    else:
        return [signal[start:stop], False]


def convert_to_fft(window_start, window_end, window_step, channel,
                   fft_min_freq, fft_max_freq, sampling_frequency, file_path,parameters):
    """
    Split an interval into 1 second intervals with 0.5 second overlap, applying FFT.

    parmas:
    window_start - the beginning of interval in seconds.
    window_end - the end of interval in seconds.  if window_end == -1 then till the end of file
    window_step - the overlap in seconds.
    channel -
    fft_min_freq - the min frequency.
    fft_max_freq - the max ferquency.
    sampling_ferquency - the frequency of the edf file.
    file_path - the path to the edf file.

    return:
    np.array - interval splitted into 1 secons segments with 0.5 second overlap, with FFT applied.
    """
    pipeline = Pipeline([FFT(), Slice(fft_min_freq, fft_max_freq), Magnitude(), Log10()])

    start, step = int(np.floor(window_start * sampling_frequency)), int(np.floor(window_step * sampling_frequency))
    stop = start + step
    # if window_end == -1:
    #     stop = -1

    lst = file_path.split('/')
    file_name = lst[-1][:-4]
    fft_data = []

    montage = str(parameters.loc['montage']['value'])
    montage_list = re.split(';', montage)
    # print("montage_list: ", montage_list)
    # print("montage_list[channel]: ", montage_list[channel])
    electrode_list = re.split('-', montage_list[channel])
    import mne
    f = mne.io.read_raw_edf(file_path)
    f.load_data()
    signal_labels = f.ch_names
    # f = pyedflib.EdfReader(file_path)
    # signal_labels = f.getSignalLabels()

    if window_end == -1:
        is_eof = False
        while not is_eof:
            [extracted_signal_from_electrode_1,is_eof_ch1] = extract_signal(f, signal_labels, electrode_list[0], start, stop)
            [extracted_signal_from_electrode_2,is_eof_ch2] = extract_signal(f, signal_labels, electrode_list[1], start, stop)
            is_eof = is_eof_ch1 or is_eof_ch2
            if is_eof:
                break

            signal_window = np.array(extracted_signal_from_electrode_1 - extracted_signal_from_electrode_2)
            fft_window = pipeline.apply(signal_window)

            fft_data.append(fft_window)
            start, stop = start + step, stop + step
    else:
        while stop <= window_end * sampling_frequency:
            [extracted_signal_from_electrode_1,is_eof] = extract_signal(f, signal_labels, electrode_list[0], start, stop)
            [extracted_signal_from_electrode_2,is_eof] = extract_signal(f, signal_labels, electrode_list[1], start, stop)

            signal_window = np.array(extracted_signal_from_electrode_1 - extracted_signal_from_electrode_2)
            fft_window = pipeline.apply(signal_window)

            fft_data.append(fft_window)
            start, stop = start + step, stop + step
            # print(fft_data)

    f._close()
    del f

    return np.array(fft_data)


def infer_model(model_path, edf_path,parameters):
    model = simple_EEGNet().to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    fname = os.path.basename(edf_path)
    # arr = convert_to_fft(29.1, 38.1, 0.5, 0, 1, 96, 250, test_file_path_edf,parameters)
    # print(f'arr: {arr},\nlen(arr): {len(arr)},\nlen(arr[0]): {len(arr[0])}')
    start = 0
    end = -1

    for channel in range(22):
        arr = convert_to_fft(start, end, 0.5, channel, 1, 96, 250, edf_path, parameters)
        with torch.no_grad():
            if arr.min() == float('-inf') or arr.max() == float('inf'):
                continue
            output = model(torch.from_numpy(arr.float().to(device)))

        Path = '/home/eshuranov/projects/eeg_epileptiform_detection/output/' + fname + '_chanel_' + str(channel) + '.csv'

        with open(Path, 'w', newline='') as output_file:
            write = csv.writer(output_file)
            write.writerow(output)
            write.writerow(output.argmax(dim = 1))
            write.writerow(np.arange(start, end, 0.5).tolist())

def montage_extract_signal(f, channel, start, stop):
    """
    f - opened edf file.
    channel - montage
    start - start of the window in seconds.
    stop - end of the window in seconds.
    """
    # signal = np.array(f.readSignal(channel))
    signal = f._data[:, channel]
    if start > len(signal) or stop > len(signal):
        return [signal, True]
    else:
        return [signal[start:stop], False]

def montage_convert_to_fft(window_start, window_end, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path):
    """
    Split an interval into 1 second intervals with 0.5 second overlap, applying FFT.

    parmas:
    window_start - the beginning of interval in seconds.
    window_end - the end of interval in seconds.  if window_end == -1 then till the end of file
    window_step - the overlap in seconds.
    channel - is already some montage chanel
    fft_min_freq - the min frequency.
    fft_max_freq - the max ferquency.
    sampling_ferquency - the frequency of the edf file.
    file_path - the path to the edf file.

    return:
    np.array - interval splitted into 1 secons segments with 0.5 second overlap, with FFT applied.
    """
    pipeline = Pipeline([FFT(), Slice(fft_min_freq, fft_max_freq), Magnitude(), Log10()])

    f = pyedflib.EdfReader(file_path)
    signal_labels = f.getSignalLabels()
    fft_by_chanels = []

    for ch in range(len(signal_labels)):
        start, step = int(np.floor(window_start * sampling_frequency)), int(np.floor(window_step * sampling_frequency))
        stop = start + step
        if window_end == -1:
            fft_data = []
            is_eof = False
            while not is_eof:
                [extracted_signal_from_electrode_1,is_eof] = montage_extract_signal(f, ch, start, stop)
                if is_eof:
                    break

                signal_window = np.array(extracted_signal_from_electrode_1)
                fft_window = pipeline.apply(signal_window)

                fft_data.append(fft_window)
                start, stop = start + step, stop + step
        else:
            fft_data = []
            while stop <= window_end * sampling_frequency:
                [extracted_signal_from_electrode_1,is_eof] = montage_extract_signal(f, ch, start, stop)

                signal_window = np.array(extracted_signal_from_electrode_1)
                fft_window = pipeline.apply(signal_window)

                fft_data.append(fft_window)
                start, stop = start + step, stop + step
        print("ch:", ch)
        fft_by_chanels.append(fft_data)
    f._close()
    del f

    return np.array(fft_by_chanels)


def montage_infer_model(model_path, edf_path):
    model = simple_EEGNet().to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    fname = os.path.basename(edf_path)
    # arr = convert_to_fft(29.1, 38.1, 0.5, 0, 1, 96, 250, test_file_path_edf,parameters)
    # print(f'arr: {arr},\nlen(arr): {len(arr)},\nlen(arr[0]): {len(arr[0])}')
    start = 0
    end = -1

    arr = montage_convert_to_fft(start, end, 0.5, 1, 96, 500, edf_path)
    for channel in range(len(arr)):
        results = []
        with torch.no_grad():
            # for n_frame in range(len(arr)):
            if arr[channel].min() == float('-inf') or arr[channel].max() == float('inf'):
                continue
                # torch.from_numpy(np.array(data[sel_row])).float()
            output = model(torch.from_numpy(arr[channel]).float().to(device))
            # print('output: {}\t'.format(output))
            # results.append(output)


        Path = '/home/eshuranov/projects/eeg_epileptiform_detection/output/' + fname + '_chanel_' + str(
            channel) + '.csv'

        with open(Path, 'w', newline='') as output_file:
            write = csv.writer(output_file)
            write.writerow(output)
            write.writerow(output.argmax(dim = 1))
            if end == -1:
                write.writerow(np.arange(start, len(arr)/2, 0.5).tolist())
            else:
                write.writerow(np.arange(start, end, 0.5).tolist())



if __name__ == '__main__':
    print("infer started")
    # parameters = pd.read_csv('/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/parameters.csv',
    #                          index_col=['parameter'])
    parameters = pd.read_csv('/home/eshuranov/projects/eeg_epileptiform_detection/data/mbt sample/parameters_mtb.csv',
                             index_col=['parameter'])
    # test_file_path_edf = '/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/TUEV/tuev/edf/train/aaaaablw/aaaaablw_00000001.edf'
    # test_file_path_edf = '/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/TUEV/tuev/edf/train/aaaaaabs/aaaaaabs_00000001.edf'
    # test_file_path_edf = '/home/eshuranov/projects/eeg_epileptiform_detection/data/mbt sample/sample mbt.edf'
    test_file_path_edf = '/home/eshuranov/projects/eeg_epileptiform_detection/data/sample hospital/1.1ictal.edf'

    model_path ='/home/eshuranov/projects/eeg_epileptiform_detection1719_step1720_of_1216_loss_196.19630297645926.pt'
    # infer_model(model_path, test_file_path_edf,parameters)

    montage_infer_model(model_path, test_file_path_edf)

