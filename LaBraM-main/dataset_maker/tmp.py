
def mergeNeighbouringEvents(events_tensor, min_duration_between_events):
    """
    Объединение соседних событий, разделённых промежутком меньше min_duration_between_events.
    Используется мягкое решение с функцией sigmoid для аппроксимации дискретного порога.
    """
    if events_tensor.shape[0] == 0 or events_tensor.shape[1] == 0:
        return events_tensor

    # Предполагаем, что события уже отсортированы по времени начала
    merged_events = []
    current_event = events_tensor[0]

    for i in range(1, events_tensor.shape[0]):
        next_event = events_tensor[i]
        gap = next_event[0] - current_event[1]

        # Мягкое решение: если gap < min_duration_between_events, sigmoid ≈ 1
        merge_decision = torch.sigmoid(100 * (gap - min_duration_between_events))

        # Плавное объединение событий, без использования явных условий
        new_end = merge_decision * torch.maximum(current_event[1], next_event[1]) + \
                  (1 - merge_decision) * current_event[1]

        # Плавно комбинируем события с коэффициентом, который определяется merge_decision
        current_event = torch.stack([current_event[0], new_end])

        # Добавляем текущее событие, так как оно уже объединено или готово к следующей итерации
        merged_events.append(current_event)

    return torch.stack(merged_events, dim=0)



def mergeNeighbouringEvents(events_tensor, min_duration_between_events):
    """
    МЯГКОЕ объединение соседних событий, БЕЗ изменения числа строк на выходе
    и БЕЗ дискретных if, зависящих от значений тензора.

    Выход: тензор той же формы [N, 2],
    где каждая строка - "обновлённое" событие.
    """

    # Если совсем нет событий (N=0), просто вернём пустой.
    # Обычно такая проверка "не ломает" дифференцируемость, т.к.
    # это ветвление не зависит от числовых значений (глобальный случай).
    if events_tensor.shape[0] == 0:
        return events_tensor

    # Гарантируем float (на всякий случай).
    events_tensor = events_tensor.float()

    # Инициализируем выход тем же размером.
    # Сразу скопируем все события как "по умолчанию".
    # shape: [N, 2]
    merged_tensor = events_tensor.clone()

    # Проходим по событиям от второго до последнего
    for i in range(1, events_tensor.shape[0]):
        current_event = merged_tensor[i - 1]  # уже "обновлённая" (мягко) версия предыдущего
        next_event = merged_tensor[i]  # текущее событие

        gap = next_event[0] - current_event[1]

        # Мягкое решение: если gap < min_duration_between_events, sigmoid ≈ 1 => "сливаем"
        # Если gap >> min_duration_between_events, sigmoid -> 0 => "не сливаем"
        merge_decision = torch.sigmoid(100.0 * (gap - min_duration_between_events))

        # Вычислим новую "правую границу" для нашего события i
        # (текущего), как плавную смесь:
        # Если merge_decision ~ 1, берём max(current_event[1], next_event[1])
        # Если ~ 0, оставляем как есть (т.е. "не двигаем" правую границу).
        new_end = merge_decision * torch.maximum(current_event[1], next_event[1]) \
                  + (1.0 - merge_decision) * next_event[1]

        # Запишем новое (мягко) "объединённое" событие в i-ю строку.
        # Левая граница у i-го события не меняется (можно менять, если хочется "слить сильнее").
        # Правую заменяем new_end.
        merged_tensor[i-1, 1] = new_end

    return merged_tensor




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
    hyp_event = mergeNeighbouringEvents(hyp_event[0], minDurationBetweenEvents)
    ref_event = splitLongEvents(ref_event, maxEventDuration)
    hyp_event = splitLongEvents(hyp_event, maxEventDuration)

    numSamples = int(end_pos - start_pos)

    ref_event = extendEvents(ref_event, toleranceStart, toleranceEnd, numSamples, fs)

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



def build_full_partition_with_limits(
        events_tensor: torch.Tensor,
        end_of_timeline: float
) -> torch.Tensor:
    """
    Формирует тензор промежутков [M, 2], покрывающих от 0 до end_of_timeline,
    учитывая отсортированные события [N, 2].

    - Если events_tensor.shape[0] == 0, вернёт один интервал [0, end_of_timeline].
    - Иначе добавляет 0 и end_of_timeline к набору границ (starts/ends событий),
      убирает дубликаты и формирует интервалы подряд.
    """
    # Если вообще нет событий, просто один интервал [0, end_of_timeline].
    if events_tensor.shape[0] == 0:
        return torch.tensor([[0.0, end_of_timeline]], device=events_tensor.device , dtype=events_tensor.dtype)

    # events_tensor: [N, 2], каждое событие [start_i, end_i].
    starts = events_tensor[:, 0]
    ends = events_tensor[:, 1]

    # Собираем все ключевые точки (граничные моменты):
    #  - 0 (начало всей шкалы),
    #  - начала всех событий,
    #  - концы всех событий,
    #  - end_of_timeline (конец всей шкалы).
    boundary_points = torch.cat([
        torch.tensor([0.0], device=events_tensor.device, dtype=events_tensor.dtype),
        starts,
        ends,
        torch.tensor([end_of_timeline], device=events_tensor.device, dtype=events_tensor.dtype)
    ], dim=0)

    # Сортируем
    boundary_points, _ = torch.sort(boundary_points)
    # Убираем повторяющиеся
    boundary_points = torch.unique(boundary_points)

    # Теперь boundary_points — это отсортированные уникальные моменты времени
    # от 0 до end_of_timeline включительно.
    # Формируем интервалы, беря попарно соседние точки.
    left = boundary_points[:-1]
    right = boundary_points[1:]

    intervals = torch.stack([left, right], dim=1)  # shape [M, 2]

    return intervals





def mergeNeighbouringEvents(events_tensor, min_duration):
    """
    Мягкое объединение соседних событий без изменения количества строк.
    - events_tensor: тензор формы [N, 2], где каждая строка — [start, end].
    - min_duration: скаляр, порог, при gap < min_duration события "сливаются" (мягко).

    Возвращает тензор той же формы [N, 2], где интервалы скорректированы.
    Градиенты беспрепятственно проходят, так как нет дискретных if по значениям.
    """
    # Если нет вообще событий, вернуть как есть (обычно это не проблема для дифференцируемости,
    # так как это ветвление по "размеру батча", а не по значениям).
    if events_tensor.shape[0] == 0:
        return events_tensor

    # Копируем, чтобы не затирать входной тензор
    merged_tensor = events_tensor.clone()

    # Проходим по событиям слева направо (от i=1 до N-1)
    for i in range(1, merged_tensor.shape[0]):
        prev_event = merged_tensor[i - 1]  # [start_{i-1}, end_{i-1}]
        curr_event = merged_tensor[i]  # [start_i, end_i]

        gap = curr_event[0] - prev_event[1]

        # "Сила слияния": если gap << min_duration => merge_decision ~ 1, иначе ~ 0
        merge_decision = torch.sigmoid(100.0 * (min_duration - gap))

        # -- Вычислим "объединённые" границы --
        # Левая граница (опционально) можно тоже сближать:
        union_left = torch.minimum(prev_event[0], curr_event[0])
        union_right = torch.maximum(prev_event[1], curr_event[1])

        # Плавно «притягиваем» левую границу текущего события к union_left
        # (если merge_decision ≈ 1, значит хотим "слиться" с предыдущим)
        new_left = merge_decision * union_left + (1 - merge_decision) * curr_event[0]

        # Аналогично плавно «растягиваем» правую границу к union_right
        new_right = merge_decision * union_right + (1 - merge_decision) * prev_event[1]

        # Записываем обратно в текущую строку
        merged_tensor[i, 0] = new_left
        merged_tensor[i-1, 1] = new_right

    return merged_tensor


def get_randomized_sample_range(sz_events, max_batch_size, fs):
    # Выбираем случайное событие
    selected_event_index = random.choice(sz_events)

    # Определяем диапазон выборки вокруг выбранного события
    center_position = (selected_event_index[0] + selected_event_index[1]) // 2
    half_window_size = max_batch_size * fs // 2

    # Случайная вариация от центра события
    variation = random.randint(-half_window_size // 2, half_window_size // 2)
    start_pos = max(center_position + variation - half_window_size, 0)
    end_pos = min(start_pos + max_batch_size * fs, len(sz_events))

    # Обновление позиций, если они выходят за границы
    if end_pos - start_pos < max_batch_size * fs:
        diff = max_batch_size * fs - (end_pos - start_pos)
        start_pos = max(start_pos - diff // 2, 0)
        end_pos = min(end_pos + diff // 2, len(sz_events))

    return start_pos, end_pos


def train_class_batch(model, samples, target, criterion, ch_names, max_batch_size):
    fs = 200
    n_samples = samples.shape[-1]
    sz_events = labram_events_to_sz_events(target[0], 0, n_samples / fs, is_labram_events_in_min=False)

    # Получение списка всех индексов начала событий
    event_indices = [(int(start.item() * fs), int(end.item() * fs)) for start, end in sz_events]

    # Если нет событий, возвращаемся без потерь
    if len(event_indices) == 0:
        return torch.tensor(0.0).to(samples.device), None, None

    start_pos, end_pos = get_randomized_sample_range(event_indices, max_batch_size, fs)

    smple_5_sec = resample_tensor(samples[0, :, start_pos:end_pos], new_freq=1)
    smple_5_sec = rearrange(smple_5_sec, 'B N (A T) -> B N A T', T=200).float().to(samples.device, non_blocking=True)
    ref = labram_events_to_sz_events(target[0], start_pos / fs, end_pos / fs).to(samples.device)

    outputs = model(smple_5_sec, input_chans=ch_names)
    hyp = (outputs.softmax(dim=1))[:, 0:3].sum(dim=1)
    f1, sens = f1_sz_estimation(hyp, ref)

    add_loss_for_f1_sz_estimation = torch.where(
        torch.isnan(f1),
        torch.tensor(1.0, device=samples.device),
        1 - f1
    ).to(samples.device)

    ref_mask = mask_from_events(ref, smple_5_sec.shape[0], fs)
    return add_loss_for_f1_sz_estimation, outputs, ref_mask



#  Предыдущая версия -------------------------------------------------------------------------------------------------------------------------------

def train_class_batch(model, samples, target, criterion, ch_names, max_batch_size):
    """
    Основная функция обучения на батче.
    Для обеспечения дифференцируемости вместо дискретных операций (например, argmax) используются softmax и мягкие аппроксимации.
    """
    fs = 200
    n_samples = samples.shape[-1]
    if n_samples > max_batch_size * fs:
        start_pos = torch.randint(n_samples - max_batch_size * fs, (1,), device=samples.device)
        end_pos = start_pos + max_batch_size * fs
    else:
        start_pos = torch.tensor(0, device=samples.device)
        end_pos = torch.tensor(n_samples, device=samples.device)
    smple_5_sec = resample_tensor(samples[0, :, start_pos:end_pos], new_freq=1)
    smple_5_sec = rearrange(smple_5_sec, 'B N (A T) -> B N A T', T=200).float().to(samples.device, non_blocking=True)
    ref = labram_events_to_sz_events(target[0], start_pos / fs, end_pos / fs).to(samples.device)

    outputs = model(smple_5_sec, input_chans=ch_names)
    hyp = (outputs.softmax(dim=1))[:, 0:3].sum(dim=1)
    f1, sens = f1_sz_estimation(hyp, ref)

    add_loss_for_f1_sz_estimation = torch.where(
        torch.isnan(f1),
        torch.tensor(1.0, device=samples.device),
        1 - f1
    ).to(samples.device)

    ref_mask = mask_from_events(ref, smple_5_sec.shape[0], fs)
    return add_loss_for_f1_sz_estimation, outputs, ref_mask
# ----------------------------------------------------------------------------------------------------------------------------------------------------