# --------------------------------------------------------

# --------------------------------------------------------
import mne
import numpy as np
import os
import pickle

from debugpy.launcher import channel
from tqdm import tqdm
from pathlib import Path
from epilepsy2bids.annotations import Annotations

SzCORE_format = ['Fp1-Avg', 'F3-Avg', 'C3-Avg', 'P3-Avg', 'O1-Avg', 'F7-Avg', 'T3-Avg', 'T5-Avg', 'Fz-Avg', 'Cz-Avg', 'Pz-Avg', 'Fp2-Avg', 'F4-Avg', 'C4-Avg', 'P4-Avg', 'O2-Avg', 'F8-Avg', 'T4-Avg', 'T6-Avg']

labram_event_types = ['SPSW', 'PLED' , 'GPED', 'EYEM', 'ARTF', 'BCKG']
#  Siena: {'sz_foc_ia': 33, 'sz_foc_f2b': 10, 'sz_foc_a': 4}   TUSZ: {'sz': 3971, 'bckg': 5969}           CHB-MIT:{'sz': 198, 'bckg': 545}  - no need
event_Types = ['sz_foc_a', 'sz_foc_f2b', 'sz_foc_ia','bckg','sz'] # [PLED , GPED, PLED]
# output:
# Siena {2.0: 37, 3.0: 10}

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

def find_index_with_substring(lst, substring):
    try:
        return next(i for i, item in enumerate(lst) if substring in item)
    except StopIteration:
        return -1

def readEDF(fileName, ref_tsv):

    events_list = []
    eventData = Annotations.loadTsv(ref_tsv).events

    for ev in eventData:
        ch_ind = find_index_with_substring(ch_names_after_convert_sz_chalenge_2025_montage, ev["channels"][0])
        if ev["eventType"].value == event_Types[0]:
            eventType = 2   # PLED
        elif ev["eventType"].value == event_Types[1]:
            eventType = 3  # GPED
        elif ev["eventType"].value == event_Types[2]:
            eventType = 2  # PLED
        elif ev["eventType"].value == event_Types[3]:
            eventType = 6
        elif ev["eventType"].value == event_Types[4]:
            eventType = 1    # SPSW
        else:
            eventType = -1

        if not ((eventType == -1) or (eventType >3)):
            ev_list = [ch_ind, float(ev["onset"]),float(ev["duration"]),eventType]
            events_list.append(ev_list)

        # ev_list = [ch_ind, float(ev["onset"])/60,float(ev["duration"]),ev["eventType"].value]
        # events_list.append(ev_list)

    if len(events_list) == 0:
        print(eventData, "  in file: ", fileName,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return [None, None, None, None]

    # return [None, None, np.array(events_list), None]

    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    # if Rawdata.ch_names != SzCORE_format:   # there are a lot of mistakes in chanels names in SzCORE with small / big latters. if ignore - it is ok
    #     raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    if signals.shape[1] < 1000:
        print("len(signals): ", len(signals), "  in file: ", fileName,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return [None, None, None, None]

    Rawdata.close()
    return [signals, times,  np.array(events_list), Rawdata]


def load_up_objects(BaseDir, OutDir):
    ev_type_dict={}
    n_none_events = 0 # none events or short file < 5 sec
    # for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
    for subject in Path(BaseDir).glob("sub-*"):
        print("Found directory: %s" % subject)
        for ref_tsv in subject.glob("**/*.tsv"):  # use tsv for loop to be sure that we will have
            print(ref_tsv)
            edf_path = ref_tsv
            fname = str(edf_path)[:-10]+'eeg.edf'   #  replace "events.tsv" to "eeg.edf"
            if not os.path.exists(fname):
                print("*.tsv file without *.edf : ",ref_tsv)
                continue

            last_part = os.path.basename(ref_tsv)
            res_pkl_name = os.path.join(OutDir,last_part)[:-4]+'.pkl'


            print("\t%s" % fname)

            try:
                [signals, times, event, Rawdata] = readEDF( fname, ref_tsv)  # event is the .rec file in the form of an array
                if event is None:
                    n_none_events += 1
                    continue
                for ev in list(event):
                    if ev_type_dict.get(ev[3]) is not None:
                        ev_type_dict[ev[3]] += 1
                    else:
                        ev_type_dict[ev[3]] = 1
                # continue
                # if banana_half_montage:
                #     signals = convert_signals_half_banana(signals, Rawdata)
                # elif sz_chalenge_2025_montage:
                #     signals = convert_signals_sz_chalenge_2025_montage(signals, Rawdata)
                # else:
                #     signals = convert_signals_sz_chalenge_2025_montage(signals, Rawdata)  # debug

            except (ValueError, KeyError):
                print("something funky happened in " + dirName + "/" + fname)
                continue
            # signals, offending_channels, labels = BuildEvents(signals, times, event)

            # for idx, (signal, event_i) in enumerate(
            #     zip(signals, event)
            # ):
            sample = {
                "signal": signals,
                "event": event,
            }
            save_pickle(sample, res_pkl_name)
    print(ev_type_dict, "n_none_events or short file < 5 sec:", n_none_events)
    return

def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


"""
 convert to labram pkl
"""

banana_half_montage = False
sz_chalenge_2025_montage = False  # no need convert, it is already in correct montage
is_random_val = False

# root = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena"
# out_path = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_to_labram_pkl"
#
# root = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_CHB-MIT"
# out_path = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_CHB-MIT_to_labram_pkl"

root = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess"
out_path = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess_pkl"

all_out_dir = os.path.join(out_path, "all")

if not os.path.exists(all_out_dir):
    os.makedirs(all_out_dir)

fs = 200

load_up_objects(
    root, all_out_dir
)


#transfer to train and eval
seed = 4523
np.random.seed(seed)

All_files = os.listdir(all_out_dir)

train_sub = list(set([f.split("_")[0] for f in All_files]))
print("train sub", len(train_sub))

if is_random_val:
    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.2), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in All_files if f.split("_")[0] in val_sub]
    train_files = [f for f in All_files if f.split("_")[0] in train_sub]
else:
    num_of_val = int(len(train_sub)*0.2)
    val_sub = train_sub[:num_of_val]
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in All_files if f.split("_")[0] in val_sub]
    train_files = [f for f in All_files if f.split("_")[0] in train_sub]

    train_path = os.path.join(out_path, 'splited', 'train')
    eval_path  = os.path.join(out_path, 'splited', 'eval')

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    for file in train_files:
        os.system(f"cp {os.path.join(all_out_dir, file)} {train_path}")
    for file in val_files:
        os.system(f"cp {os.path.join(all_out_dir, file)} {eval_path}")







