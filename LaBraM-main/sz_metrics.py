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


def labram_events_to_sz_events(labram_events, start_pos, end_pos, is_labram_events_in_min=False):
    """
    Преобразование событий LabRAM в формат Sz с использованием векторизованных операций.
    Здесь используются torch.clamp и булевое индексирование для обеспечения дифференцируемости.
    """
    device = labram_events.device
    # Предполагается, что labram_events имеет форму [N, 4]
    if is_labram_events_in_min:  # не бывает минут в TUEV Siena TUSZ,  остальные стоит проверить
        event_start = labram_events[:, 1] * 60
        event_end = (labram_events[:, 2] + labram_events[:, 1]) * 60
    else:
        event_start = labram_events[:, 1]
        event_end = (labram_events[:, 2] + labram_events[:, 1])

    event_start = torch.clamp(event_start, min=start_pos)
    event_end = torch.clamp(event_end, max=end_pos)
    event_start = event_start - start_pos
    event_end = event_end - start_pos

    events = torch.stack([event_start, event_end], dim=1)
    valid_mask = (event_end - event_start) > 0
    events = events[valid_mask]

    # Оставляем только события с ID 1, 2, 3
    allowed_ids = [1, 2, 3]
    id_mask = torch.tensor([id in allowed_ids for id in labram_events[valid_mask, 3]], dtype=torch.bool, device=device)
    filtered_events = events[id_mask]

    # Проверяем наличие других событий
    other_event_mask = (~id_mask) & (
        ~torch.tensor([id in [4, 5, 6] for id in labram_events[valid_mask, 3]], dtype=torch.bool, device=device))
    if torch.any(other_event_mask):
        if (4 not in set(labram_events[valid_mask][other_event_mask][:, 3].tolist()) and
            5 not in set(labram_events[valid_mask][other_event_mask][:, 3].tolist()) and
            6 not in set(labram_events[valid_mask][other_event_mask][:, 3].tolist())):
            print("Обнаружены другие события:", set(labram_events[valid_mask][other_event_mask][:, 3].tolist()))

    # Добавляем сортировку по времени начала событий
    sorted_indices = torch.argsort(filtered_events[:, 0])
    filtered_events = filtered_events[sorted_indices]

    # Удаляем дубликаты
    unique_events = list(set(map(tuple, filtered_events.cpu().tolist())))
    unique_events.sort(key=lambda x: x[0])

    return torch.tensor(unique_events, device=device)


def events_from_mask(data, fs):
    """
    Построение событий из бинарной маски с использованием дифференцируемой аппроксимации.
    Вместо жёсткого порога используется свёртка и мягкая функция активации (sigmoid).
    """
    # Приводим данные к float и добавляем размерности: [1, 1, T]
    data = data.float().unsqueeze(0).unsqueeze(0)

    # Считаем производную сигнала через свертку
    kernel = torch.tensor([[-1, 1]], dtype=torch.float32, device=data.device).unsqueeze(0)
    diff = F.conv1d(data, kernel, padding=0).squeeze()  # shape: [T-1]

    # Мягко определяем моменты переходов
    start_scores = torch.sigmoid(10 * diff)  # Высокое значение при положительном переходе
    end_scores = torch.sigmoid(-10 * diff)  # Высокое значение при отрицательном переходе

    # Определяем пороговые значения для обнаружения переходов
    threshold = 0.5
    starts = (start_scores > threshold).nonzero(as_tuple=False).flatten()
    ends = (end_scores > threshold).nonzero(as_tuple=False).flatten()

    # Формируем события (интервалы), проверяя наличие парных начал и концов
    events = []
    i = 0
    while i < len(starts) and i < len(ends):
        if starts[i] <= ends[i]:
            events.append((starts[i], ends[i]))
            i += 1
        else:
            print(f'Warning: Start {starts[i]} is after End {ends[i]}, skipping')
            i += 1

    # Преобразуем список событий в тензор
    if events:
        events_tensor = torch.tensor(events, dtype=torch.float32, device=data.device).unsqueeze(0)
    else:
        events_tensor = torch.empty(0, 2, dtype=torch.float32, device=data.device).unsqueeze(0)

    return events_tensor


import torch


def mergeNeighbouringEvents(events_tensor: torch.Tensor,
                            min_duration_between_events: float) -> torch.Tensor:
    """
    Объединение соседних событий, если разрыв между ними меньше min_duration_between_events.
    Работает с дискретной логикой (не дифференцируемо).

    Параметры:
      events_tensor: тензор формы [N,2],
                     каждая строка - [start, end].
                     Предполагается, что start <= end и события отсортированы по возрастанию start.
      min_duration_between_events: float, минимальный разрыв между событиями,
                                  при котором они считаются раздельными.

    Возвращает:
      merged_tensor: тензор [K,2], K <= N, набор объединённых событий.
    """
    # Если событий нет или одно, возвращаем как есть
    if events_tensor.shape[0] <= 1:
        return events_tensor.clone()

    merged_events = []
    # Начинаем с первого события
    current_event = events_tensor[0].clone()

    # Идём по остальным
    for i in range(1, events_tensor.shape[0]):
        next_event = events_tensor[i]

        # gap - разрыв между текущим событием и следующим
        gap = next_event[0] - current_event[1]

        if gap < min_duration_between_events:
            # Сливаем (объединяем) два события
            current_event[1] = max(current_event[1].item(), next_event[1].item())
        else:
            # Разрыв достаточно большой, добавляем текущее событие в результат
            merged_events.append(current_event)
            current_event = next_event.clone()

    # Добавляем последнее событие
    merged_events.append(current_event)

    # Формируем итоговый тензор
    return torch.stack(merged_events, dim=0)


def splitLongEvents(events_tensor: torch.Tensor, max_event_duration: float) -> torch.Tensor:
    """
    Разбиение событий, длительность которых превышает max_event_duration,
    на более короткие. Возвращает новый тензор [K, 2], где K >= N.

    Параметры:
      events_tensor: тензор [N, 2], каждая строка (start, end).
                     Предполагается, что start <= end.
      max_event_duration: максимальная допустимая длительность события (float).

    Логика:
      Для каждого события, пока длительность > max_event_duration,
      "отрезаем" кусок длины max_event_duration, добавляем его в список,
      затем двигаем start. Если кусок получился короче порога, добавляем
      оставшуюся часть и переходим к следующему событию.
    """
    if events_tensor.shape[0] == 0:
        return events_tensor  # Пустой тензор, ничего делить не нужно

    splitted = []

    for i in range(events_tensor.shape[0]):
        start = float(events_tensor[i, 0])
        end = float(events_tensor[i, 1])

        # Пока не исчерпали событие
        while start < end:
            # Конец кусочка - либо start + max_event_duration, либо сам end, если остался отрезок меньше порога
            segment_end = min(start + max_event_duration, end)

            splitted.append([start, segment_end])

            start = segment_end  # сдвигаем "начало" на конец текущего кусочка

    # Преобразуем список в тензор
    splitted_tensor = torch.tensor(splitted, dtype=events_tensor.dtype, device=events_tensor.device)
    return splitted_tensor


def extendEvents(events_tensor, before, after, num_samples):
    """
    Расширение каждого события на before секунд до начала и after секунд после окончания.
    Используются torch.clamp для обеспечения корректного диапазона.
    """
    file_duration = num_samples
    extended_events = []
    for event in events_tensor:
        new_start = torch.clamp(event[0] - before, min=0)
        new_end = torch.clamp(event[1] + after, max=file_duration)
        extended_events.append(torch.stack([new_start, new_end]))
    if len(extended_events) == 0:   # расширяем только референсные события - на них градиенты не нужны
        return torch.empty((0, 2), dtype=events_tensor.dtype, device=events_tensor.device)
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


def build_antievent_intervals(events_tensor: torch.Tensor,
                              end_of_timeline: float
                             ) -> torch.Tensor:
    """
    Возвращает тензор [M,2], где каждая строка — интервал в [0, end_of_timeline],
    НЕ пересекающийся ни с одним интервалом из events_tensor.

    Предполагается, что:
      - events_tensor.shape == [N,2],
      - каждая строка [start_i, end_i],
      - start_i <= end_i (хотя бы формально),
      - интервалы (start_i, end_i) могут частично пересекаться или выходить
        за [0, end_of_timeline].

    Шаги:
      1) Если N=0 (нет событий), весь [0, end_of_timeline] — "анти-событие".
      2) Прижимаем (clamp) все интервалы к [0, end_of_timeline].
      3) Сливаем пересекающиеся интервалы (union).
      4) Берём дополнение:
         - от 0 до начала первого интервала (если >0),
         - промежутки между концами и началами соседних интервалов,
         - от конца последнего интервала до end_of_timeline (если >0).
    """

    # Если нет событий, всё время — анти-событие
    if events_tensor.shape[0] == 0:
        # если end_of_timeline <= 0, тогда вообще нет "положительной" оси
        if end_of_timeline > 0:
            return torch.tensor([[0.0, end_of_timeline]],
                                dtype=events_tensor.dtype)
        else:
            # на всякий случай вернём пустой
            return torch.empty((0,2), dtype=events_tensor.dtype)

    # Шаг 1: Клонируем и "прижимаем" интервалы к [0, end_of_timeline]
    events_clamped = events_tensor.clone()
    events_clamped[:, 0].clamp_(0, end_of_timeline)
    events_clamped[:, 1].clamp_(0, end_of_timeline)

    # Шаг 2: сортируем по возрастанию start, если вдруг не отсортированы
    # (Если уверены, что на входе уже отсортировано, можно пропустить)
    sorted_idx = torch.argsort(events_clamped[:, 0])
    events_clamped = events_clamped[sorted_idx]

    # Шаг 3: слияние (merge) пересекающихся интервалов
    merged = []
    current_start = events_clamped[0, 0].item()
    current_end   = events_clamped[0, 1].item()

    for i in range(1, events_clamped.shape[0]):
        s = events_clamped[i, 0].item()
        e = events_clamped[i, 1].item()

        if s <= current_end:
            # Интервалы пересекаются или соприкасаются => расширяем текущий
            current_end = max(current_end, e)
        else:
            # Прерывание => запоминаем "слитый" и начинаем новый
            merged.append([current_start, current_end])
            current_start = s
            current_end   = e

    # Добавляем последний накопленный интервал
    merged.append([current_start, current_end])
    # Превращаем в тензор
    merged_events = torch.tensor(merged, dtype=events_clamped.dtype)

    # Шаг 4: строим дополнение (анти-интервалы) в [0, end_of_timeline]
    anti = []

    # 4.1: от 0 до начала первого, если это > 0
    if merged_events[0, 0] > 0:
        anti.append([0.0, merged_events[0, 0].item()])

    # 4.2: промежутки между (end_i, start_{i+1})
    for i in range(len(merged_events) - 1):
        left_end = merged_events[i, 1].item()
        right_start = merged_events[i+1, 0].item()
        if right_start > left_end:
            anti.append([left_end, right_start])

    # 4.3: от конца последнего интервала до end_of_timeline
    last_end = merged_events[-1, 1].item()
    if last_end < end_of_timeline:
        anti.append([last_end, end_of_timeline])

    # Если "дыр" нет, вернём пустой тензор
    if len(anti) == 0:
        return torch.empty((0, 2), dtype=events_clamped.dtype)

    return torch.tensor(anti, dtype=events_clamped.dtype)




def f1_sz_estimation(hyp, ref_event):
    """
    Мягкий подсчёт:
      - tp: сумма "есть ли срабатывание" на каждом из N событийных интервалов
             (реализовано как максимум гипа на интервале),
      - fp: сумма "есть ли срабатывание" на каждом из M анти-интервалов,
      - ref_true = N как тензор.
      - Оценка F1 и чувствительности для событий (например, судорожных) с использованием мягких аппроксимаций.
    Без if по данным, чтобы не ломать дифференцируемость.
    """
    toleranceStart = 30
    toleranceEnd = 60
    # minOverlap = 0.0
    maxEventDuration = 5 * 60
    minDurationBetweenEvents = 90
    device = hyp.device

    ref_event = mergeNeighbouringEvents(ref_event, minDurationBetweenEvents)
    ref_event = splitLongEvents(ref_event, maxEventDuration)
    numSamples = hyp.shape[0]
    ref_event = extendEvents(ref_event, toleranceStart, toleranceEnd, numSamples)

    ref_antievent = build_antievent_intervals(ref_event,numSamples).to(device)
    ref_antievent = splitLongEvents(ref_antievent, maxEventDuration)

    # N = ref_event.shape[0]
    # M = ref_antievent.shape[0]

    ref_true = torch.tensor(float(ref_event.shape[0]), device=device, dtype=hyp.dtype)
    t_index = torch.arange(numSamples, device=device).view(1, -1)  # shape [1, T]
    # Подготовим start_i и end_i для каждого из N интервалов shape [N,1].
    start_i = ref_event[:, 0].view(-1, 1)  # shape [N,1]
    end_i   = ref_event[:, 1].view(-1, 1)  # shape [N,1]

    membership_event = (
        (start_i <= t_index) & (t_index < end_i)  # shape [N, T], bool
    ).float()  # превращаем в 0/1
    # То же самое для анти-интервалов [M,2].
    start_j = ref_antievent[:, 0].view(-1, 1)  # shape [M,1]
    end_j   = ref_antievent[:, 1].view(-1, 1)  # shape [M,1]
    membership_antievent = (
        (start_j <= t_index) & (t_index < end_j)
    ).float()  # shape [M, T]

    presence_event = (membership_event * hyp.unsqueeze(0)).max(dim=1).values  # shape [N]
    presence_antievent = (membership_antievent * hyp.unsqueeze(0)).max(dim=1).values  # shape [M]

    # 3) tp = сумма presence_event[i], fp = сумма presence_antievent[j].
    tp_value = presence_event.sum()   # скаляр
    fp_value = presence_antievent.sum()  # скаляр

    sensitivity, precision, f1 = computeScores(ref_true, tp_value, fp_value)
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



