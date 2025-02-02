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

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']


ch_names_after_convert_sz_chalenge_2025_montage = ['FP1-Avg', 'F3-Avg', 'C3-Avg', 'P3-Avg',
                                                   'O1-Avg',  'F7-Avg', 'T3-Avg', 'T5-Avg',
                                                   'FZ-Avg',  'CZ-Avg', 'PZ-Avg', 'FP2-Avg',
                                                   'F4-Avg',  'C4-Avg', 'P4-Avg', 'O2-Avg',
                                                   'F8-Avg',  'T4-Avg', 'T6-Avg']
                                                    # Chanels names from sz chalenge 2025 discription:
                                                    # Fp1-Avg, F3-Avg, C3-Avg, P3-Avg,
                                                    # O1-Avg, F7-Avg, T3-Avg, T5-Avg,
                                                    # Fz-Avg, Cz-Avg, Pz-Avg, Fp2-Avg,
                                                    # F4-Avg, C4-Avg, P4-Avg, O2-Avg,
                                                    # F8-Avg, T4-Avg, T6-Avg
                                                    # names from debug test pipeline of sz chalenge 2025  CZ and PZ have big Z like in chOrder_standard from Labram!!
                                                    # ['FP1-Avg', 'F3-Avg', 'C3-Avg', 'P3-Avg',
                                                    # 'O1-Avg', 'F7-Avg', 'T3-Avg', 'T5-Avg',
                                                    # 'FZ-Avg', 'CZ-Avg', 'PZ-Avg', 'FP2-Avg',
                                                    # 'F4-Avg', 'C4-Avg', 'P4-Avg', 'O2-Avg',
                                                    # 'F8-Avg', 'T4-Avg', 'T6-Avg']

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
    ch_names_after_convert = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
                        'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2']

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
        )
    )  # 21
    return new_signals

def convert_signals_sz_chalenge_2025_montage(signals, Rawdata):
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

    ch_names_after_convert_sz_chalenge_2025_montage = ['FP1-Avg', 'F3-Avg', 'C3-Avg', 'P3-Avg',
                                                       'O1-Avg', 'F7-Avg', 'T3-Avg', 'T5-Avg',
                                                       'FZ-Avg', 'CZ-Avg', 'PZ-Avg', 'FP2-Avg',
                                                       'F4-Avg', 'C4-Avg', 'P4-Avg', 'O2-Avg',
                                                       'F8-Avg', 'T4-Avg', 'T6-Avg']
    Avg = (signals[signal_names["EEG FP1-REF"]]+ signals[signal_names["EEG F3-REF"]] + signals[signal_names["EEG C3-REF"]] + signals[signal_names["EEG P3-REF"]] +
           signals[signal_names["EEG O1-REF"]] + signals[signal_names["EEG F7-REF"]] + signals[signal_names["EEG T3-REF"]] + signals[signal_names["EEG T5-REF"]] +
           signals[signal_names["EEG FZ-REF"]] + signals[signal_names["EEG CZ-REF"]] + signals[signal_names["EEG PZ-REF"]] + signals[signal_names["EEG FP2-REF"]] +
           signals[signal_names["EEG F4-REF"]] + signals[signal_names["EEG C4-REF"]] + signals[signal_names["EEG P4-REF"]] + signals[signal_names["EEG O2-REF"]] +
           signals[signal_names["EEG F8-REF"]] + signals[signal_names["EEG T4-REF"]] + signals[signal_names["EEG T6-REF"]]) / 19


    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]] - Avg,  # 0
            (signals[signal_names["EEG F3-REF"]] - Avg),  # 1
            (signals[signal_names["EEG C3-REF"]] - Avg),  # 2
            (signals[signal_names["EEG P3-REF"]] - Avg),  # 3
            (signals[signal_names["EEG O1-REF"]] - Avg),  # 4
            (signals[signal_names["EEG F7-REF"]] - Avg),  # 5
            (signals[signal_names["EEG T3-REF"]] - Avg),  # 6
            (signals[signal_names["EEG T5-REF"]] - Avg),  # 7
            (signals[signal_names["EEG FZ-REF"]] - Avg),  # 8
            (signals[signal_names["EEG CZ-REF"]] - Avg),  # 9
            (signals[signal_names["EEG PZ-REF"]] - Avg),  # 10
            (signals[signal_names["EEG FP2-REF"]] - Avg),  # 11
            (signals[signal_names["EEG F4-REF"]] - Avg),  # 12
            (signals[signal_names["EEG C4-REF"]] - Avg),  # 13
            (signals[signal_names["EEG P4-REF"]] - Avg),  # 14
            (signals[signal_names["EEG O2-REF"]] - Avg),  # 15
            (signals[signal_names["EEG F8-REF"]] - Avg),  # 16
            (signals[signal_names["EEG T4-REF"]] - Avg),  # 17
            (signals[signal_names["EEG T6-REF"]] - Avg),  # 18
        )
    )
    return new_signals

def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir, banana_half_montage, sz_chalenge_2025_montage):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                    if banana_half_montage:
                        signals = convert_signals_half_banana(signals, Rawdata)
                    elif sz_chalenge_2025_montage:
                        signals = convert_signals_sz_chalenge_2025_montage(signals, Rawdata)

                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue
                signals, offending_channels, labels = BuildEvents(signals, times, event)

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
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


"""
TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
"""
banana_half_montage = False
sz_chalenge_2025_montage = True

# root = "/share/TUEV/"
# root = "/userhome1/jiangweibang/Datasets/TUH_Event/v2.0.0/edf"
root = "/media/public/Datasets/TUEV/tuev/edf"

if banana_half_montage:
    train_out_dir = os.path.join(root, "processed_train_banana_half")
    eval_out_dir = os.path.join(root, "processed_eval_banana_half")
elif sz_chalenge_2025_montage:
    train_out_dir = os.path.join(root, "processed_train_sz_chalenge_2025_montage")
    eval_out_dir = os.path.join(root, "processed_eval_sz_chalenge_2025_montage")

#
# if not os.path.exists(train_out_dir):
#     os.makedirs(train_out_dir)
# if not os.path.exists(eval_out_dir):
#     os.makedirs(eval_out_dir)
#
# BaseDirTrain = os.path.join(root, "train")
# fs = 200
# TrainFeatures = np.empty(
#     (0, 23, fs)
# )  # 0 for lack of intialization, 22 for channels, fs for num of points
# TrainLabels = np.empty([0, 1])
# TrainOffendingChannel = np.empty([0, 1])
# load_up_objects(
#     BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir, banana_half_montage, sz_chalenge_2025_montage
# )
#
# BaseDirEval = os.path.join(root, "eval")
# fs = 200
# EvalFeatures = np.empty(
#     (0, 23, fs)
# )  # 0 for lack of intialization, 22 for channels, fs for num of points
# EvalLabels = np.empty([0, 1])
# EvalOffendingChannel = np.empty([0, 1])
# load_up_objects(
#     BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir, banana_half_montage, sz_chalenge_2025_montage
# )


#transfer to train, eval, and test
seed = 4523
np.random.seed(seed)

if banana_half_montage:
    train_files = os.listdir(os.path.join(root, "processed_train_banana_half"))
    test_files = os.listdir(os.path.join(root, "processed_eval_banana_half"))
elif sz_chalenge_2025_montage:
    train_files = os.listdir(os.path.join(root, "processed_train_sz_chalenge_2025_montage"))
    test_files = os.listdir(os.path.join(root, "processed_eval_sz_chalenge_2025_montage"))


train_sub = list(set([f.split("_")[0] for f in train_files]))
print("train sub", len(train_sub))

val_sub = np.random.choice(train_sub, size=int(
    len(train_sub) * 0.2), replace=False)
train_sub = list(set(train_sub) - set(val_sub))
val_files = [f for f in train_files if f.split("_")[0] in val_sub]
train_files = [f for f in train_files if f.split("_")[0] in train_sub]

if banana_half_montage:
    train_path = os.path.join(root, 'processed_banana_half', 'processed_train_banana')
    eval_path = os.path.join(root, 'processed_banana_half', 'processed_eval_banana')
    test_path = os.path.join(root, 'processed_banana_half', 'processed_test_banana')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for file in train_files:
        os.system(f"cp {os.path.join(root, 'processed_train_banana_half', file)} {train_path}")
    for file in val_files:
        os.system(f"cp {os.path.join(root, 'processed_train_banana_half', file)} {eval_path}")
    for file in test_files:
        os.system(f"cp {os.path.join(root, 'processed_eval_banana_half', file)} {test_path}")
elif sz_chalenge_2025_montage:
    train_path = os.path.join(root, 'processed_sz_chalenge_2025_montage', 'processed_train_sz_chalenge_2025_montage')
    eval_path = os.path.join(root, 'processed_sz_chalenge_2025_montage', 'processed_eval_sz_chalenge_2025_montage')
    test_path = os.path.join(root, 'processed_sz_chalenge_2025_montage', 'processed_test_sz_chalenge_2025_montage')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for file in train_files:
        os.system(f"cp {os.path.join(root, 'processed_train_sz_chalenge_2025_montage', file)} {train_path}")
    for file in val_files:
        os.system(f"cp {os.path.join(root, 'processed_train_sz_chalenge_2025_montage', file)} {eval_path}")
    for file in test_files:
        os.system(f"cp {os.path.join(root, 'processed_eval_sz_chalenge_2025_montage', file)} {test_path}")
