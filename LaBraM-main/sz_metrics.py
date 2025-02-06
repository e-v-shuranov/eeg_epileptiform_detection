import numpy as np
import torch

def computeScores(refTrue,tp,fp):
    """ Compute performance metrics."""
    # Sensitivity
    if refTrue > 0:
        sensitivity = tp / refTrue
    else:
        sensitivity = np.nan  # no ref event

    # Precision
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan  # no hyp event

    # F1 Score
    if np.isnan(sensitivity) or np.isnan(precision):
        f1 = np.nan
    elif (sensitivity + precision) == 0:  # No overlap ref & hyp
        f1 = 0
    else:
        f1 = 2 * sensitivity * precision / (sensitivity + precision)

    # # FP Rate
    # fpRate = fp / (numSamples / fs / 3600 / 24)  # FP per day

    return sensitivity, precision, f1

def events_from_mask(data, fs):
    """
    data NDArray[Bool]:  binary vector where positive labels are indicated by True.
    fs (int): Sampling frequency in Hertz of the annotations.
    """
    d1 = torch.zeros(10)
    d2 = torch.ones(10)
    events = list()
    tmpEnd = []
    # Find transitions
    start_i = (data[1:].int()- data[:-1].int() == 1).int().nonzero()   #.squeeze()
    end_i = (data[1:].int()- data[:-1].int() == -1).int().nonzero()    #.squeeze()

    # No transitions and first sample is positive -> event is duration of file
    if start_i.shape[0] == 0 and start_i.shape[0] == 0 and data[0]:
        events.append((0, len(data) / fs))
    else:
        # Edge effect - First sample is an event
        if data[0]:
            events.append((torch.tensor(0).to(end_i.device), (end_i[0] + 1) / fs))
            end_i = end_i.index_select(0, torch.arange(1, end_i.size(0)).to(end_i.device))
                # np.delete(end_i, 0)
        # Edge effect - Last event runs until end of file
        if data[-1]:
            if len(start_i):
                tmpEnd = [((start_i[-1] + 1) / fs, torch.tensor(len(data)).to(end_i.device) / fs)]
                start_i = start_i.index_select(0, torch.arange(0, start_i.size(0) - 1).to(start_i.device))
                # start_i = np.delete(start_i, len(start_i) - 1)
        # Add all events
        start_i += 1
        end_i += 1
        for i in range(len(start_i)):
            events.append((start_i[i] / fs, end_i[i] / fs))
        events += tmpEnd  # add potential end edge effect

    return events


def mask_from_events(events, numSamples, fs):
    """
    Build binary mask associated with list of events
    """
    mask = np.zeros((numSamples,), dtype=np.bool_)
    for event in events:
        mask[round(event[0] * fs):round(event[1] * fs)] = True
    return mask

def mergeNeighbouringEvents(unsorted_events, minDurationBetweenEvents: float):
    """Merge events separated by less than longer than minDurationBetweenEvents.
    Args:
        events - list of events to split
        minDurationBetweenEvents (float): minimum duration between events [seconds]

    Returns:
        list of splited events
    """

    events = sorted(unsorted_events, key=lambda x: x[0])
    mergedEvents = events.copy()

    i = 1
    while i < len(mergedEvents):
        event = mergedEvents[i]
        if event[0] - mergedEvents[i - 1][1] < minDurationBetweenEvents:
            mergedEvents[i - 1] = (mergedEvents[i - 1][0], event[1])
            del mergedEvents[i]
            i -= 1
        i += 1

    return mergedEvents


def splitLongEvents(events, maxEventDuration: float):
    """Split events longer than maxEventDuration in shorter events.
    Args:
        events (list): containing events to split
        maxEventDuration (float): maximum duration of an event [seconds]

    Returns:
        list: Returns a new list with all events split to
            a maximum duration of maxEventDuration.
    """

    shorterEvents = events.copy()

    for i, event in enumerate(shorterEvents):
        if event[1] - event[0] > maxEventDuration:
            shorterEvents[i] = (event[0], event[0] + maxEventDuration)
            shorterEvents.insert(i + 1, (event[0] + maxEventDuration, event[1]))

    return shorterEvents


def extendEvents(events, before: float, after: float, numSamples, fs):
    """Extend duration of all events in an Annotation object.

    Args:
        events (list): containing events to extend
        before (float): Time to extend before each event [seconds]
        after (float):  Time to extend after each event [seconds]
        numSamples (int): number of samples in file (or batch)
        fs (int): Sampling frequency in Hertz of the annotations.

    Returns:
        list: Returns a new list with all events extended
    """

    extendedEvents = events.copy()
    fileDuration = numSamples / fs

    for i, event in enumerate(extendedEvents):
        extendedEvents[i] = (max(torch.tensor(0), event[0] - before), (min(torch.tensor(fileDuration), event[1] + after)))

    return extendedEvents

def labram_events_to_sz_events(labram_events, start_pos, end_pos):
    """

    Args:
        labram_events: [chanel, event_start_pos (in minutes), event_end_pos (in minutes), event (6 classes 0 1 2 Seizures  3 4 5 not Seizures)]
        start_pos: start of selected interval from signal in case if file too long (in seconds)
        end_pos:   end of selected interval from signal in case if file too long (in seconds)

    Returns:

    """
    result_list = []
    for i in range(labram_events.shape[0]):
        event_start = labram_events[i, 1] * 60 - start_pos
        event_end = labram_events[i, 2] * 60 - start_pos

        if event_end <0 or event_end > end_pos:
            continue
        if event_start < 0:
            event_start = 0
        if event_end > end_pos:
            event_end = end_pos
        result_tuple = (event_start, event_end)
        result_list.append(result_tuple)
    return result_list

def f1_sz_estimation(hyp,ref,start_pos, end_pos, fs):
    """
       ref and hyp are batch_size arrays of True/False values.
       in this function we assumed that batch consists of 64 sequentially seconds
       so we could add loss based on events, if an event exist here
    """
    # fs = 1 # here Sampling frequency is 1 sec
    toleranceStart = 30
    toleranceEnd = 60
    minOverlap = 0
    maxEventDuration = 5 * 60
    minDurationBetweenEvents = 90

    # result = scoring.EventScoring(ref, hyp)
    # ref_event = events_from_mask(ref, fs)  # ref already in sz format
    ref_event = ref
    hyp_event = events_from_mask(hyp, fs)

    # Merge events separated by less than param.minDurationBetweenEvents
    ref_event = mergeNeighbouringEvents(ref_event, minDurationBetweenEvents)
    hyp_event = mergeNeighbouringEvents(hyp_event, minDurationBetweenEvents)

    # Split long events to param.maxEventDuration
    ref_event = splitLongEvents(ref_event, maxEventDuration)
    hyp_event = splitLongEvents(hyp_event, maxEventDuration)

    refTrue = len(ref_event)
    # fp and tn  - no need

    # Count True detections
    tp = 0
    numSamples = int(end_pos - start_pos)
    tpMask = torch.zeros(numSamples)
    extendedRef = extendEvents(ref_event, toleranceStart, toleranceEnd, numSamples, fs)
    for event in extendedRef:
        relativeOverlap =  (hyp[int(event[0] * fs):int(event[1] * fs)].sum() / fs) / (event[1] - event[0])
            #
            # (np.sum(np.array(hyp[int(event[0] * fs):int(event[1] * fs)])) / fs
            #                ) / (event[1] - event[0])


        if relativeOverlap > minOverlap + 1e-6:
            tp += 1
            tpMask[int(event[0] * fs):int(event[1] * fs)] = 1

    # Count False detections
    fp = 0
    for event in hyp_event:
        if bool(torch.all(~tpMask[int(event[0] * fs):int(event[1] * fs)].bool()).item()):
            fp += 1

    sensitivity, precision, f1 = computeScores(refTrue,tp,fp)

    return f1, sensitivity