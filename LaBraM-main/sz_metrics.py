import numpy as np
import torch

import torch.nn.functional as F
from einops import rearrange


def resample_tensor(tensor, new_freq=1):
    """
    Дифференцируемое ресэмплирование входного тензора.
    Некоторые индексации вычисляются как целые числа – они не влияют на поток градиента, так как вычисления проводятся на исходном тензоре.
    """
    old_freq = 200
    channels, time_len = tensor.shape
    dt_old = 1 / old_freq
    dt_new = 1 / new_freq
    new_time_len = int(time_len * dt_old // dt_new)
    slices = []
    window_start = -int(2 * old_freq)
    window_end = int(3 * old_freq)
    target_width = int(5 * old_freq)
    for i in range(new_time_len):
        t_center = int(i * dt_new / dt_old)
        start_idx = max(window_start + t_center, 0)
        end_idx = min(window_end + t_center, time_len)
        if start_idx == 0:
            pad_left = target_width - (end_idx - start_idx)
            padded_slice = torch.cat([tensor[:, :pad_left], tensor[:, start_idx:end_idx]], dim=1)
        elif end_idx == time_len:
            pad_right = target_width - (end_idx - start_idx)
            padded_slice = torch.cat([tensor[:, start_idx:end_idx], tensor[:, -pad_right:]], dim=1)
        else:
            padded_slice = tensor[:, start_idx:end_idx]
        slices.append(padded_slice)
    return torch.stack(slices, dim=0)

def labram_events_to_sz_events(labram_events, start_pos, end_pos):
    """
    Преобразование событий LabRAM в формат Sz с использованием векторизованных операций.
    Здесь используются torch.clamp и булевое индексирование для обеспечения дифференцируемости.
    """
    # Предполагается, что labram_events имеет форму [N, 4]
    event_start = labram_events[:, 1] * 60 - start_pos
    event_end = labram_events[:, 2] * 60 - start_pos
    event_start = torch.clamp(event_start, min=0)
    event_end = torch.clamp(event_end, max=end_pos)
    events = torch.stack([event_start, event_end], dim=1)
    valid_mask = (event_end - event_start) > 0
    events = events[valid_mask]
    return events

def events_from_mask(data, fs):
    """
    Построение событий из бинарной маски с использованием дифференцируемой аппроксимации.
    Вместо жёсткого порога используется свёртка и мягкая функция активации (sigmoid).
    """
    # Приводим данные к float и добавляем размерности: [1, 1, T]
    data = data.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[-1, 1]], dtype=torch.float32, device=data.device).unsqueeze(0)
    diff = F.conv1d(data, kernel, padding=0).squeeze(0).squeeze(0)  # shape: [T-1]
    # Мягкое обнаружение переходов
    start_scores = torch.sigmoid(10 * diff)  # высокое значение при положительном переходе
    end_scores = torch.sigmoid(10 * (-diff))  # высокое значение при отрицательном переходе
    T = data.shape[-1]
    indices = torch.arange(1, T, device=data.device, dtype=torch.float32)
    # Вычисляем «средневзвешенные» индексы начала и конца (мягкая аппроксимация)
    start_index = torch.sum(indices * start_scores) / (torch.sum(start_scores) + 1e-6)
    end_index = torch.sum(indices * end_scores) / (torch.sum(end_scores) + 1e-6)
    events = torch.stack([start_index, end_index]).unsqueeze(0)
    return events

def mergeNeighbouringEvents(events_tensor, min_duration_between_events):
    """
    Объединение соседних событий, разделённых промежутком меньше min_duration_between_events.
    Используется мягкое решение с функцией sigmoid для аппроксимации дискретного порога.
    """
    if events_tensor.shape[0] == 0:
        return events_tensor
    # Предполагаем, что события уже отсортированы по времени начала
    merged_events = []
    current_event = events_tensor[0]
    for i in range(1, events_tensor.shape[0]):
        next_event = events_tensor[i]
        gap = next_event[0] - current_event[1]
        # Мягкое решение: если gap < min_duration_between_events, sigmoid ≈ 1
        merge_decision = torch.sigmoid(100 * (min_duration_between_events - gap))
        new_end = merge_decision * torch.maximum(current_event[1], next_event[1]) + (1 - merge_decision) * \
                  current_event[1]
        if merge_decision > 0.5:
            current_event = torch.stack([current_event[0], new_end])
        else:
            merged_events.append(current_event)
            current_event = next_event
    merged_events.append(current_event)
    return torch.stack(merged_events, dim=0)


def splitLongEvents(events_tensor, max_event_duration):
    """
    Разбиение событий, длительность которых превышает max_event_duration, на более короткие с использованием
    дифференцируемого вычисления нового начала каждого под-события.
    """
    if events_tensor.shape[0] == 0:
        return events_tensor
    split_events = []
    for event in events_tensor:
        duration = event[1] - event[0]
        # Вычисляем количество разбиений. Используем torch.ceil для получения количества шагов.
        num_splits = torch.ceil(duration / max_event_duration)
        num_splits_int = int(num_splits.item())

        # Если число разбиений равно 1, возвращаем событие как есть.
        if num_splits_int <= 1:
            split_events.append(torch.stack([event[0], event[1]]))
        else:
            # Вычисляем новый интервал, используя дифференцируемое линейное пространство от 0 до 1.
            new_starts = event[0] + (event[1] - max_event_duration - event[0]) * \
                         torch.linspace(0, 1, steps=num_splits_int,
                                        device=events_tensor.device, dtype=event[0].dtype)
            # Для каждого нового начала вычисляем конец под-события
            for ns in new_starts:
                new_start = ns
                new_end = torch.minimum(new_start + max_event_duration, event[1])
                split_events.append(torch.stack([new_start, new_end]))
    return torch.stack(split_events, dim=0)

def extendEvents(events_tensor, before, after, num_samples, fs):
    """
    Расширение каждого события на before секунд до начала и after секунд после окончания.
    Используются torch.clamp для обеспечения корректного диапазона.
    """
    file_duration = num_samples / fs
    extended_events = []
    for event in events_tensor:
        new_start = torch.clamp(event[0] - before, min=0)
        new_end = torch.clamp(event[1] + after, max=file_duration)
        extended_events.append(torch.stack([new_start, new_end]))
    return torch.stack(extended_events, dim=0)

def computeScores(ref_true, tp, fp):
    """
    Вычисление метрик (чувствительность, точность, F1) с использованием дифференцируемых операций.
    Здесь torch.where используется для выбора ветки, однако ветки с константами не будут передавать градиенты.
    """
    sensitivity = torch.where(ref_true > 0, tp / ref_true, torch.tensor(0.0, device=ref_true.device))
    precision = torch.where((tp + fp) > 0, tp / (tp + fp), torch.tensor(0.0, device=ref_true.device))
    f1 = torch.where((sensitivity + precision) > 0,
                     2 * sensitivity * precision / (sensitivity + precision),
                     torch.tensor(0.0, device=sensitivity.device))
    return sensitivity, precision, f1

def f1_sz_estimation(hyp, ref, start_pos, end_pos, fs):
    """
    Оценка F1 и чувствительности для событий (например, судорожных) с использованием мягких аппроксимаций.
    Вместо дискретного подсчёта используются дифференцируемые маски, построенные по времени.
    """
    toleranceStart = 30
    toleranceEnd = 60
    minOverlap = 0.0
    maxEventDuration = 5 * 60
    minDurationBetweenEvents = 90
    # Предполагаем, что ref уже содержит события в формате [start, end]
    ref_event = ref
    hyp_event = events_from_mask(hyp, fs)
    ref_event = mergeNeighbouringEvents(ref_event, minDurationBetweenEvents)
    hyp_event = mergeNeighbouringEvents(hyp_event, minDurationBetweenEvents)
    ref_event = splitLongEvents(ref_event, maxEventDuration)
    hyp_event = splitLongEvents(hyp_event, maxEventDuration)

    numSamples = int(end_pos - start_pos)

    # Вместо передачи start_pos и end_pos напрямую в torch.linspace, создаём равномерный вектор u от 0 до 1
    u = torch.linspace(0, 1, steps=numSamples, device=hyp.device, dtype=start_pos.dtype)
    time_axis = start_pos + (end_pos - start_pos) * u

    def build_soft_mask(events, time_axis):
        mask = torch.zeros_like(time_axis)
        for event in events:
            # Мягкий индикатор события через произведение сигмоидов.  Сигмоида с высоким коэффициентом (100) аппроксимирует дискретный переход.
            mask += torch.sigmoid(100 * (time_axis - event[0])) * (1 - torch.sigmoid(100 * (time_axis - event[1])))
        return torch.clamp(mask, 0, 1)

    ref_mask = build_soft_mask(ref_event, time_axis)
    hyp_mask = build_soft_mask(hyp_event, time_axis)

    # Вычисляем true positive как сумму минимальных значений двух масок
    tp_val = torch.sum(torch.minimum(ref_mask, hyp_mask))
    fp_val = torch.sum(hyp_mask) - tp_val
    ref_true = torch.sum(ref_mask)

    sensitivity, precision, f1 = computeScores(ref_true, tp_val, fp_val)
    return f1, sensitivity

def mask_from_events(events, numSamples, fs):
    """
    Построение бинарной маски по списку событий с использованием дифференцируемой аппроксимации.
    Здесь используется сигмоида для мягкого приближения индикаторной функции.
    """
    time_axis = torch.linspace(0, numSamples / fs, steps=numSamples, device=events.device)
    mask = torch.zeros_like(time_axis)
    for event in events:
        mask += torch.sigmoid(100 * (time_axis - event[0])) * (1 - torch.sigmoid(100 * (time_axis - event[1])))
    # В качестве финального шага можно применить пороговую функцию (не дифференцируемую), или вернуть мягкую маску
    return mask > 0.5



