# --------------------------------------------------------
# mbt data preprocessing
# --------------------------------------------------------
import mne
import numpy as np
import os
import pickle
from tqdm import tqdm


# drop_channels may be we have to enlarge
drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

mbt_drop_channels = ['Status', 'Fz-Cz', 'Cz-Pz']
# mbt_chOrder_standard = ['Fp2-F8', 'F8 - T2', 'T2 - T4', 'T4 - T6', 'T6-O2', 'Fp1-F7', 'F7 - T1', 'T1 - T3', 'T3-T5', 'T5-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fz-Cz', 'Cz-Pz']
mbt_chOrder_standard = ['Fp1-F7', 'F7 - T1', 'T1 - T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8 - T2', 'T2 - T4', 'T4 - T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

# clinic_standart0 = ['EEG Fp2-F8', 'EEG F8-T2', 'EEG T2-T4', 'EEG T4-T6', 'EEG T6-O2', 'EEG Fp1-F7', 'EEG F7-T1', 'EEG T1-T3', 'EEG T3-T5', 'EEG T5-O1', 'EEG Fp2-F4', 'EEG F4-C4', 'EEG C4-P4', 'EEG P4-O2', 'EEG Fp1-F3', 'EEG F3-C3', 'EEG C3-P3', 'EEG P3-O1', 'EEG Fz-Cz', 'EEG Cz-Pz', 'ECG EKG']
clinic_drop_channels = ['EEG Fz-Cz', 'EEG Cz-Pz', 'ECG EKG']
clinic_standart = ['EEG Fp1-F7','EEG F7-T1','EEG T1-T3','EEG T3-T5','EEG T5-O1', 'EEG Fp2-F8','EEG F8-T2','EEG T2-T4','EEG T4-T6', 'EEG T6-O2','EEG Fp1-F3', 'EEG F3-C3', 'EEG C3-P3','EEG P3-O1','EEG Fp2-F4', 'EEG F4-C4', 'EEG C4-P4','EEG P4-O2']

new_ch_names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3",
                "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]   # it is correct but LaBraM may be have an error , have to check
# new_ch_names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3",
#                 "P3-O1", "FP2-F4", "F4-C4", "C4-P4"]



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
    signals = np.concatenate([signals, signals, signals], axis=1)  # We take a 5-second interval around the event and ring the boundaries. [s (for ring),s (real),s (for ring)]
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
    # clinic_standart => new_ch_names

    # clinic_standart =
    #['EEG Fp1-F7', 'EEG F7-T1', 'EEG T1-T3', 'EEG T3-T5', 'EEG T5-O1',
    # 'EEG Fp2-F8', 'EEG F8-T2', 'EEG T2-T4', 'EEG T4-T6', 'EEG T6-O2',
    # 'EEG Fp1-F3', 'EEG F3-C3', 'EEG C3-P3', 'EEG P3-O1',
    # 'EEG Fp2-F4', 'EEG F4-C4', 'EEG C4-P4']

    # new_ch_names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    #                 "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    #                 "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    #                 "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]

    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG Fp1-F7"]]
        ,   # 0  "FP1-F7"
            (
                signals[signal_names["EEG F7-T1"]]
                + signals[signal_names["EEG T1-T3"]]
            ),  # 1   "F7-T7"   == "F7-T3"
            (
                signals[signal_names["EEG T3-T5"]]
            ),  # 2   "T7-P7"  ==  "T3-T5"
            (
                signals[signal_names["EEG T5-O1"]]
            ),  # 3   "P7-O1"  == "T5-O1"

            (
                signals[signal_names["EEG Fp2-F8"]]
            ),  # 4   "FP2-F8"
            (
                signals[signal_names["EEG F8-T2"]]
                + signals[signal_names["EEG T2-T4"]]
            ),  # 5   "F8-T8"  == "F8-T4"
            (
                signals[signal_names["EEG T4-T6"]]
            ),  # 6   "T8-P8"  == "T4-T6"
            (
                signals[signal_names["EEG T6-O2"]]
            ),  # 7   "P8-O2"
            (
                signals[signal_names["EEG Fp1-F3"]]
            ),  # 8   "FP1-F3"
            (
                signals[signal_names["EEG F3-C3"]]
            ),  # 9   "F3-C3"
            (
                signals[signal_names["EEG C3-P3"]]
            ),  # 10`   "C3-P3"
            (
                signals[signal_names["EEG P3-O1"]]
            ),  # 11   "P3-O1"
            (
                signals[signal_names["EEG Fp2-F4"]]
            ),  # 12   "FP2-F4"
            (
                signals[signal_names["EEG F4-C4"]]
            ),  # 13   "F4-C4"
            (
                signals[signal_names["EEG C4-P4"]]
            )  # 14   "C4-P4"
            ,(
                signals[signal_names["EEG P4-O2"]]
            )  # 15   "P4-O2"     "ch+1" bug in LaBraM pretrain  ,  "P4-O2" - out of bounds
        )
    )
    return new_signals, new_ch_names


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if clinic_drop_channels is not None:
        useless_chs = []
        for ch in clinic_drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if clinic_standart is not None and len(clinic_standart) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(clinic_standart)
    if Rawdata.ch_names != clinic_standart:
        raise ValueError

    if Rawdata.times.shape[0] < 8449:
        # q1 = mne.io.Raw.copy(Rawdata)
        # q2 = mne.io.Raw.copy(Rawdata)
        # q = mne.io.Raw.copy(mne.concatenate_raws([q1,q2]))

        # # q = mne.io.Raw.copy(mne.concatenate_raws([Rawdata, Rawdata,Rawdata,Rawdata,Rawdata,Rawdata]))
        # q.filter(l_freq=0.1, h_freq=75.00)
        # print(q.times.shape[0])
        tmp =  mne.io.Raw.copy(Rawdata)
        while Rawdata.times.shape[0] < 8449:
            Rawdata = mne.concatenate_raws([Rawdata, tmp])
        mne.io.Raw.save(Rawdata, "/media/public/Datasets/sample_hospital/processed/qqq_raw.fif",overwrite=True)
        Rawdata = mne.io.read_raw_fif("/media/public/Datasets/sample_hospital/processed/qqq_raw.fif", preload=True)
    Rawdata.filter(l_freq=0.1, h_freq=75.00)
    Rawdata.notch_filter(50.0)
    # Rawdata.filter(l_freq=0.1, h_freq=75.0, filter_length = 1500)
    # Rawdata.notch_filter(50.0, filter_length = 1500)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    num_of_sec = int(times[-1]-1)

    signals, new_chanels = convert_signals(signals,Rawdata)


    if not os.path.exists(RecFile):
        eventData = np.zeros([signals.shape[0]*num_of_sec, 4])
        # eventData[0] - chanel  eventData[1] ==  start eventData[2] == end  eventData[3] == 0 label
        for ch_ in range(signals.shape[0]):
            for current_sec in range(num_of_sec):
                eventData[(ch_*num_of_sec)+current_sec,0] = ch_
                eventData[(ch_*num_of_sec)+current_sec,1] = current_sec
                eventData[(ch_*num_of_sec)+current_sec,2] = current_sec+1
    else:
        eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    # Lets check eventData format!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # RecFile = fileName[0:-3] + "rec"
    # eventData = np.genfromtxt(RecFile, delimiter=",")
    return [signals, times, eventData]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
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
clinic dataset 
"""

# root = "/userhome1/jiangweibang/Datasets/TUH_Event/v2.0.0/edf"
root = "/media/public/Datasets/sample_hospital"
eval_out_dir = os.path.join(root, "processed_data_10_09_2024")
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)

BaseDirTrain = os.path.join(root, "ictal_examples")
fs = 200
mbtFeatures = np.empty(
    (0, 23, fs)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
mbtLabels = np.empty([0, 1])
mbtOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirTrain, mbtFeatures, mbtLabels, mbtOffendingChannel, eval_out_dir
)


#transfer to eval
seed = 4523
np.random.seed(seed)


test_files = os.listdir(os.path.join(root, "processed_data_10_09_2024"))
print("clinic test", len(test_files))

for file in test_files:
    os.system(f"cp \"{os.path.join(root, 'processed_data_10_09_2024', file)}\" {os.path.join(root, 'processed', 'processed_test_ictal_examples')}")
    # os.system(f"cp {os.path.join(root, 'processed_data_10_09_2024', file)} {os.path.join(root, 'processed', 'processed_test')}")
