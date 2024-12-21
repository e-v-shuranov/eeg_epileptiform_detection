# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import mne
import numpy as np
import os
import pickle
from tqdm import tqdm

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""
# banana
# ch_names_after_convert = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
#                           'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
#                           'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
#                           'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
# half banana
# ch_names_after_convert = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
#                           'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2']

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

drop_channels.extend(['EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5', 'EEG F9', 'EEG Fc2', 'EEG Fc6', 'EEG Cp2', 'EEG Cp6', 'EEG F10', \
     'EKG EKG', 'SPO2', 'HR', '1', '2', 'EEG P9', 'EEG P10', 'B', 'C', 'D', 'PLET', '61', '62', '63', '64', 'MK'])
chOrder_standard_siena =['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG F7', \
                         'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Cz', 'EEG Pz']




def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 200.0
    [numChan, numPoints] = signals.shape
    # for i in range(numChan):  # standardize each channel
    #     if np.std(signals[i, :]) > 0:
    #         signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)

        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    ch_names_after_convert = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
                        'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]]
            - signals[signal_names["EEG F7-REF"]],  # 0
            (
                signals[signal_names["EEG F7-REF"]]
                - signals[signal_names["EEG T3-REF"]]
            ),  # 1
            (
                signals[signal_names["EEG T3-REF"]]
                - signals[signal_names["EEG T5-REF"]]
            ),  # 2
            (
                signals[signal_names["EEG T5-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 3
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F8-REF"]]
            ),  # 4
            (
                signals[signal_names["EEG F8-REF"]]
                - signals[signal_names["EEG T4-REF"]]
            ),  # 5
            (
                signals[signal_names["EEG T4-REF"]]
                - signals[signal_names["EEG T6-REF"]]
            ),  # 6
            (
                signals[signal_names["EEG T6-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  # 7
            (
                signals[signal_names["EEG FP1-REF"]]
                - signals[signal_names["EEG F3-REF"]]
            ),  # 14
            (
                signals[signal_names["EEG F3-REF"]]
                - signals[signal_names["EEG C3-REF"]]
            ),  # 15
            (
                signals[signal_names["EEG C3-REF"]]
                - signals[signal_names["EEG P3-REF"]]
            ),  # 16
            (
                signals[signal_names["EEG P3-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 17
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F4-REF"]]
            ),  # 18
            (
                signals[signal_names["EEG F4-REF"]]
                - signals[signal_names["EEG C4-REF"]]
            ),  # 19
            (
                signals[signal_names["EEG C4-REF"]]
                - signals[signal_names["EEG P4-REF"]]
            ),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  # 21
    return new_signals


def convert_signals_half_banana(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }

    # input
    # chOrder_standard_siena = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4',
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
                signals[signal_names["EEG Fp2"]]
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
        if 'EEG FP2' in Rawdata.ch_names:
            Rawdata.rename_channels({'EEG FP2': 'EEG Fp2'})
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
    for i in range(len(lines)):
        if ('PN01.edf' in lines[i] and path[-1] == 'PN01-1.edf'):
            start_time = count_seconds(lines[i + 1].split()[-1])
            end_time = count_seconds(lines[i + 2].split()[-1])

            seizure_start_time = count_seconds(lines[i + 5].split()[-1].replace(":", ".")) - start_time
            seizure_end_time = count_seconds(lines[i + 6].split()[-1]) - start_time
            seizure_arr.append([seizure_start_time, seizure_end_time])
            seizure_start_time = count_seconds(lines[i + 9].split()[-1]) - start_time
            seizure_end_time = count_seconds(lines[i + 10].split()[-1]) - start_time
            seizure_arr.append([seizure_start_time, seizure_end_time])
            print("exception for PN01.edf seizure_arr:", seizure_arr)
            break
        elif (path[-1] in lines[i] or ('PNO6-4.edf' in lines[i] and path[-1] == 'PN06-4.edf')
                or ('PNO6-2.edf' in lines[i] and path[-1] == 'PN06-2.edf')
                or ('PN11-.edf' in lines[i] and path[-1] == 'PN11-1.edf')
                or ('PNO6-1.edf' in lines[i] and path[-1] == 'PN06-1.edf')):


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


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):

    # testpath = "/media/public/Datasets/siena-scalp-eeg-database-1.0.0/PN01/PN01-1.edf"
    # try:
    #     [signals, times, event] = readEDF(
    #         testpath
    #     )
    #
    # except (ValueError, KeyError):
    #     print("!!!")
    # print(event)
    # exit(0)

    dir_list = os.listdir(BaseDir)
    for p_path in dir_list:
        p_path_full = os.path.join(BaseDir, p_path)
        for dirName, subdirList, fileList in tqdm(os.walk(p_path_full)):
            print("Found directory: %s" % dirName)
            for fname in fileList:
                if fname[-4:] == ".edf":
                    print("\t%s" % fname)
                    try:
                        [signals, times, event] = readEDF(
                            dirName + "/" + fname
                        )  # event is the .rec file in the form of an array

                    except (ValueError, KeyError):
                        print("something funky happened in " + dirName + "/" + fname)
                        continue
                    # offending_channels == -1 because we do not going to use info about chanels and labels now
                    signals, offending_channels, labels = BuildEvents(signals, times, event)

                    # store
                    p_out_dir = os.path.join(OutDir, fname[0:-4])
                    if not os.path.exists(p_out_dir):
                        os.makedirs(p_out_dir)

                    for idx, (signal, offending_channel, label) in enumerate(
                        zip(signals, offending_channels, labels)
                    ):
                        sample = {
                            "signal": signal,
                            "offending_channel": offending_channel,
                            "label": label,
                        }
                        save_pickle(
                            sample,
                            os.path.join(
                                p_out_dir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                            ),
                        )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


"""
Siena dataset is downloaded from https://XXX
"""

root = "/media/public/Datasets/siena-scalp-eeg-database-1.0.0"

all_out_dir = os.path.join(root, "processed_all_banana_half")
if not os.path.exists(all_out_dir):
    os.makedirs(all_out_dir)

BaseDirTrain = root
fs = 200
TrainFeatures = np.empty(
    (0, 23, fs)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
TrainOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, all_out_dir
)

