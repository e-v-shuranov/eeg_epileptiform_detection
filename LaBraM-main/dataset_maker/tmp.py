import mne
import os
import numpy as np
from tornado.autoreload import start

# bug with 'EEG FP2'
# chOrder_standard_siena =['EEG Fp1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG F7', \
#                          'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Cz', 'EEG Pz']


drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

drop_channels.extend(['EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5', 'EEG F9', 'EEG Fc2', 'EEG Fc6', 'EEG Cp2', 'EEG Cp6', 'EEG F10', \
     'EKG EKG', 'SPO2', 'HR', '1', '2', 'EEG P9', 'EEG P10', 'B', 'C', 'D', 'PLET', '61', '62', '63', '64', 'MK'])

chOrder_standard_siena =['EEG Fp1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG F7', \
                         'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Cz', 'EEG Pz']

def convert_signals_half_banana(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }

    # input
    # chOrder_standard_siena = ['EEG Fp1', 'EEG FP2', 'EEG F3', 'EEG F4',
    #                           'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
    #                           'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8',
    #                           'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6',
    #                           'EEG Fz', 'EEG Cz', 'EEG Pz']
    # output
    ch_names_after_convert = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
                              'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2']
    # it is equal to
    # new_ch_names_to_128 = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    #                        "FP2-F8", "F8-T8", "T8-P8", "P8-O2"]
    # because T6 == P8, T4 == T8, T3 == T7, T5 == P7
    # ch_names_after_convert - use because it is fit to sieana labels
    # new_ch_names_to_128    - we will use in Labram to fit standard_1020

    new_signals = np.vstack(
        (
            signals[signal_names["EEG Fp1"]]
            - signals[signal_names["EEG F7"]],
                # 0
            (
                signals[signal_names["EEG F7"]]
                - signals[signal_names["EEG T3"]]
            ),  # 1
            (
                signals[signal_names["EEG T3"]]
                - signals[signal_names["EEG T5"]]
            ),  # 2
            (
                signals[signal_names["EEG T5"]]
                - signals[signal_names["EEG O1"]]
            ),  # 3
            (
                signals[signal_names["EEG FP2"]]
                - signals[signal_names["EEG F8"]]
            ),  # 4
            (
                signals[signal_names["EEG F8"]]
                - signals[signal_names["EEG T4"]]
            ),  # 5
            (
                signals[signal_names["EEG T4"]]
                - signals[signal_names["EEG T6"]]
            ),  # 6
            (
                signals[signal_names["EEG T6"]]
                - signals[signal_names["EEG O2"]]
            ),  # 7
        )
    )  # 21
    return new_signals, ch_names_after_convert

def read_dir_txt(file_path):
    if not os.path.exists(file_path):
        print(f'no labels for directory {file_path}')
        return []

    print('the file was found')
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def count_seconds(string):
    string = string.split('.')

    return int(string[0]) * 3600 + int(string[1]) * 60 + int(string[2])

def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard_siena is not None and len(chOrder_standard_siena) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard_siena)
    if Rawdata.ch_names != chOrder_standard_siena:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    num_of_sec = int(times[-1] - 1)


    # print(RecFile, num_of_sec)
    path = fileName.split('/')
    txt_file_name = '/'.join(path[:-1]) + '/Seizures-list-' + path[-2] + '.txt'
    lines = read_dir_txt(txt_file_name)

    seizure_arr = []
    print(lines)
    print(path[-1])
    for i in range(len(lines)):
        if path[-1] in lines[i] or ('PNO6-4.edf' in lines[i] and path[-1] == 'PN06-4.edf'):
            print('seizure exists')
            start_time = count_seconds(lines[i + 1].split()[-1])
            end_time = count_seconds(lines[i + 2].split()[-1])

            seizure_start_time = count_seconds(lines[i + 3].split()[-1]) - start_time
            seizure_end_time = count_seconds(lines[i + 4].split()[-1]) - start_time
            seizure_arr.append([seizure_start_time, seizure_end_time])

    signals, new_channels = convert_signals_half_banana(signals, Rawdata)
    registered_num_seconds = end_time - start_time

    # print(seizure_arr)
    if not os.path.exists(RecFile):
        eventData = np.zeros([num_of_sec, 4])
        # eventData[0] - chanel  eventData[1] ==  start eventData[2] == end  eventData[3] == 0 label
        ch_ = -1
        for current_sec in range(num_of_sec):
            eventData[current_sec, 0] = ch_
            eventData[current_sec, 1] = current_sec
            eventData[current_sec, 2] = current_sec + 1

            if current_sec > registered_num_seconds:
                print(f'file name {fileName}: current second {current_sec} is no longer registered')
                continue
            else:
                for interval in seizure_arr:
                    if interval[0] <= current_sec <= interval[1]:
                        print(current_sec, interval, fileName)
                        eventData[current_sec, 3] = 1
                        break

    else:
        eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()

    return [signals, times, eventData]


file = '/media/public/Datasets/siena-scalp-eeg-database-1.0.0/PN06/PN06-4.edf'
readEDF(file)