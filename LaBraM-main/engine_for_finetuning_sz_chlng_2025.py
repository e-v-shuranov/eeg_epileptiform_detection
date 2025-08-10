# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import math
import sys
from typing import Iterable, Optional
import torch
# from pyarrow.compute import random
import random
from timm.utils import ModelEma
import utils
from einops import rearrange
import pywt
import pickle
from pyhealth.metrics import  multiclass_metrics_fn
from collections import defaultdict

# from ML_solution.infer_model import device
from sz_metrics import f1_sz_estimation, labram_events_to_sz_events, events_from_mask, mask_from_events
from sz_metrics_old import f1_sz_estimation as f1_sz_estimation_old
from sz_metrics_old import labram_events_to_sz_events as labram_events_to_sz_events_old
from sz_metrics_old import events_from_mask as events_from_mask_old
from sz_metrics_old import mask_from_events as mask_from_events_old

import numpy as np
import torch.nn.functional as F

def resample_tensor(tensor, new_freq=1):
    # Исходная частота дискретизации
    old_freq = 200

    # Размер исходного тензора
    channels , time_len  = tensor.shape

    # Время одной выборки в секундах
    dt_old = 1 / old_freq
    dt_new = 1 / new_freq

    # Количество временных шагов в новой временной шкале
    new_time_len = round(time_len * dt_old // dt_new)

    # Новый тензор для хранения результата
    new_tensor = torch.zeros((new_time_len, channels, 5*old_freq))

    # Границы интервала для каждого нового временного шага
    window_start = -round(2 * old_freq)
    window_end = round(3 * old_freq)

    for i in range(new_time_len):
        t_center = round(i * dt_new / dt_old)

        start_idx = max(window_start + t_center, 0)
        end_idx = min(window_end + t_center, time_len)

        if start_idx == 0:
            # Если начало окна выходит за пределы начала тензора,
            # используем padding
            pad_left = abs(round(5 * old_freq) - end_idx)
            padded_slice = torch.cat([tensor[:,:pad_left], tensor[:,start_idx:end_idx]], dim=1)
        elif end_idx == time_len:
            # Если конец окна выходит за пределы конца тензора,
            # используем padding
            pad_right = abs(2*time_len - round(5 * old_freq) - start_idx )
            padded_slice = torch.cat([tensor[:,start_idx:end_idx], tensor[:,pad_right:]], dim=1)
        else:
            # Если окно полностью внутри границ тензора
            padded_slice = tensor[:,start_idx:end_idx]
        new_tensor[i] = padded_slice

    return new_tensor


def get_events_based_data(samples, event_data_torch , fs=200, max_batch_size=512):
    if samples.shape[2] < 5*fs:
        print(" file less then 5 seconds")
        return None, None
    if event_data_torch.shape[1]>max_batch_size:
        print("!!!event_data_torch.shape[0]>max_batch_size!!!")
        len = max_batch_size
    else:
        len = event_data_torch.shape[1]

    events_samples = torch.zeros((len, samples.shape[1], 5*fs), dtype=torch.float)
    for i in range(len):
        start = int(((event_data_torch[0,i,1] - 2) * fs).round())
        end = start + int(5 * fs)
        if start<0:
            start = 0
            end = int(5 * fs)
        if end>samples.shape[2]:
            end = samples.shape[2]
            start = end - int(5 * fs)
        events_samples[i] = samples[0,:,start:end]

    labels = event_data_torch[0,:len, 3].to(dtype=int) - 1

    return events_samples, labels

from torch.nn.functional import one_hot, softmax, log_softmax
from torch import gather

# def one_hot_encoding(y):
#     # Используем F.one_hot для создания one-hot encoding
#     y_onehot = one_hot(y.long(), num_classes=y.size(0))
#     return y_onehot.float()

#
# def my_loss(y_pred, y_true, criterion):
#     # Получение индексов классов
#     indices = (y_pred.argmax(1) < 3).long().view(-1, 1)
#
#     # Применение softmax для нормализации вероятностей
#     y_pred_softmax = softmax(y_pred, dim=1)
#
#     # Сборка нужных значений из нормализованных вероятностей
#     y_pred_filtered = torch.gather(y_pred_softmax, 1, indices).squeeze(1)
#
#     # Преобразование к типу float
#     y_pred_filtered = y_pred_filtered.to(dtype=torch.float)
#
#     # Логарифмические вероятности для y_pred
#     log_probs = log_softmax(y_pred_filtered.unsqueeze(1), dim=-1)
#
#     # Целевые классы
#     target_indices = y_true.view(-1, 1).float()
#
#     # Потеря
#     loss = criterion(log_probs, target_indices)
#
#     return loss

# def my_loss11(y_pred, y_true, criterion):
#     """ for debug, to use crossentrapy instead of MSELoss"""
#     indices = (y_pred.argmax(1) < 3).long().view(-1, 1)
#     y_pred_softmax = softmax(y_pred, dim=1)
#     y_pred_filtered = gather(y_pred_softmax, 1, indices).squeeze(1)
#     y_pred_filtered = y_pred_filtered.to(dtype=torch.float)
#
#     log_probs = log_softmax(y_pred_filtered.unsqueeze(1), dim=-1)
#     target_indices = torch.argmax(y_true, dim=1).unsqueeze(1)
#
#     # y_true_onehot = one_hot_encoding(y_true)
#     # y_pred_onehot = one_hot_encoding(y_pred_filtered)
#
#     # logits = torch.log(y_pred_onehot + 1e-10)
#
#     return criterion(log_probs, target_indices)

def get_randomized_sample_range(sz_events, max_batch_size, signal_length, fs):
    # Выбираем случайное событие
    selected_event_index = random.choice(sz_events)

    # Определяем диапазон выборки вокруг выбранного события
    center_position = (selected_event_index[0] + selected_event_index[1]) // 2
    half_window_size = max_batch_size * fs // 2

    # Случайная вариация от центра события
    variation = random.randint(-half_window_size // 2, half_window_size // 2)
    start_pos = max(center_position + variation - half_window_size, 0)
    end_pos = min(start_pos + max_batch_size * fs, signal_length)

    # Обновление позиций, если они выходят за границы
    if end_pos - start_pos < max_batch_size * fs:
        diff = max_batch_size * fs - (end_pos - start_pos)
        start_pos = max(start_pos - diff // 2, 0)
        end_pos = min(end_pos + diff // 2, signal_length)

    return start_pos, end_pos

def train_class_batch_binary_from_6_classes(model, samples, target, criterion, ch_names, max_batch_size):
    fs = 200
    n_samples = samples.shape[-1]
    if n_samples<1000:
        samples = torch.cat([samples, torch.zeros(samples.shape[0],samples.shape[1],(1000-n_samples)).to(samples.device)], dim=2)

    events_samples, targets = get_events_based_data(samples, target, fs = fs, max_batch_size=max_batch_size)
    events_samples = rearrange(events_samples, 'B N (A T) -> B N A T', T=200).float().to(samples.device, non_blocking=True)
    outputs_for_events = model(events_samples, ch_names)
    hyp = (outputs_for_events.softmax(dim=1))[:, 0:3].sum(dim=1)
    ref = (targets < 3).long()
    loss = criterion(hyp, ref.float())

    # Если нет событий или слишком короткая запись, возвращаемся без потерь
    loss = torch.where(
        torch.tensor(len(targets) == 0 or n_samples < 1000, device=samples.device),
        torch.tensor(0.0, device=samples.device),
        loss.to(samples.device)
    )

    return loss , outputs_for_events, targets

def train_class_batch_original(model, samples, target, criterion, ch_names, max_batch_size):
    fs = 200
    n_samples = samples.shape[-1]
    if n_samples<1000:
        samples = torch.cat([samples, torch.zeros(samples.shape[0],samples.shape[1],(1000-n_samples)).to(samples.device)], dim=2)

    events_samples, targets = get_events_based_data(samples, target, fs = fs, max_batch_size=max_batch_size)
    events_samples = rearrange(events_samples, 'B N (A T) -> B N A T', T=200).float().to(samples.device, non_blocking=True)
    outputs_for_events = model(events_samples, ch_names)
    loss = criterion(outputs_for_events, targets)

    # Если нет событий или слишком короткая запись, возвращаемся без потерь
    loss = torch.where(
        torch.tensor(len(targets) == 0 or n_samples < 1000, device=samples.device),
        torch.tensor(0.0, device=samples.device),
        loss.to(samples.device)
    )

    return loss , outputs_for_events, targets

def train_class_batch(model, samples, target, criterion, ch_names, max_batch_size):
    fs = 200
    n_samples = samples.shape[-1]
    sz_events = labram_events_to_sz_events(target[0], 0, n_samples / fs, is_labram_events_in_min=False)

    # Получение списка всех индексов начала событий
    event_indices = [(int(start.item() * fs), int(end.item() * fs)) for start, end in sz_events]

    # Если нет событий, возвращаемся без потерь
    if n_samples<1000:
        samples = torch.cat([samples, torch.zeros(samples.shape[0],samples.shape[1],(1000-n_samples)).to(samples.device)], dim=2)

    start_pos, end_pos = get_randomized_sample_range(event_indices, max_batch_size, samples.shape[-1], fs)

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

    # Если нет событий или слишком короткая запись, возвращаемся без потерь
    add_loss_for_f1_sz_estimation = torch.where(
        torch.tensor(len(event_indices) == 0 or n_samples < 1000, device=samples.device),
        torch.tensor(0.0, device=samples.device),
        add_loss_for_f1_sz_estimation.to(samples.device)
    )

    ref_mask = mask_from_events(ref, (smple_5_sec.shape[0]))
    return add_loss_for_f1_sz_estimation, outputs, ref_mask




def train_class_batch1(model, samples, target, criterion, ch_names, max_batch_size):
    fs = 200
    events_samples, targets = get_events_based_data(samples, target, fs = fs, max_batch_size=max_batch_size)
    events_samples = rearrange(events_samples, 'B N (A T) -> B N A T', T=200).float().to(samples.device, non_blocking=True)
    outputs_for_events = model(events_samples, ch_names)
    # loss = criterion(outputs_for_events, targets)

    # old_fs = 200  # in samples
    # new_fs = 1   # in outputs

    n_samples = samples.shape[-1]
    if n_samples > max_batch_size * fs:
        start_pos = torch.randint(n_samples - max_batch_size * fs, (1,)).to(samples.device)
        end_pos = start_pos + max_batch_size * fs
    else:
        start_pos = torch.tensor(0).to(samples.device)
        end_pos = torch.tensor(n_samples).to(samples.device)

    smple_5_sec = resample_tensor(samples[0, :, start_pos:end_pos], new_freq=1)
    smple_5_sec = rearrange(smple_5_sec, 'B N (A T) -> B N A T', T=200).float().to(samples.device, non_blocking=True)
    ref = labram_events_to_sz_events(target[0], start_pos/fs, end_pos/fs).to(samples.device)
    outputs = model(smple_5_sec, ch_names)
    hyp = (outputs.argmax(1) < 3)
    hyp_event = events_from_mask(hyp, fs)
    f1, sens = f1_sz_estimation(hyp_event, ref, start_pos / fs, end_pos / fs, fs=1)

    add_loss_for_f1_sz_estimation = torch.where(
        torch.isnan(f1), torch.tensor(1.0, device=samples.device),  # Штраф за отсутствие детекции
        1 - f1  # Основное значение ошибки
    )
    loss = add_loss_for_f1_sz_estimation.to(samples.device)

    ref_mask = mask_from_events(ref, smple_5_sec.shape[0], fs)
    # hyp_mask = mask_from_events(hyp_event, smple_5_sec.shape[0], fs)
    # from torch.nn import MSELoss, CrossEntropyLoss
    # cr1 = MSELoss()
    # loss = my_loss(outputs, ref, cr1)
    # # return torch.norm(outputs_for_events[:,0] - targets), outputs_for_events, targets
    # return loss, (outputs_for_events.argmax(1) < 3)[:2].to(dtype=torch.float), targets[:2]

    alpha = 1

    return loss, outputs, ref_mask
    # return loss + alpha * add_loss_for_f1_sz_estimation, outputs_for_events, targets


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch_sz_chlng_2025(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True, max_batch_size=512):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    # num_training_steps_per_epoch = 2
    for data_iter_step, (samples, fname, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print("start file:",fname)   # debug
        if ("Siena" in fname[0]) or ("tuh_train" in fname[0]):
            TUSZ_or_Siena = True
        else:
            TUSZ_or_Siena = False
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True) / 100
        # samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)

        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)
        alpha = 0.5
        if loss_scaler is None:
            samples = samples.half()
            loss, output, target_events = train_class_batch(
                model, samples, targets, criterion, input_chans, max_batch_size)
        else:
            with torch.cuda.amp.autocast():
                if step % 2 == 0:
                    if TUSZ_or_Siena:
                        loss, output, target_events = train_class_batch(
                            model, samples, targets, criterion, input_chans, max_batch_size)
                        # loss, output, target_events = train_class_batch_binary_from_6_classes(
                        #     model, samples, targets, torch.nn.BCEWithLogitsLoss(), input_chans, max_batch_size)
                    else:
                        loss, output, target_events = train_class_batch_original(
                            model, samples, targets, criterion, input_chans, max_batch_size)
                    loss = alpha * loss
                else:
                    loss, output, target_events = train_class_batch(
                        model, samples, targets, criterion, input_chans, max_batch_size)
        # if loss == None:
        #     loss, output, target_events =torch.empty(1, 1, device=samples.device), torch.empty(1, device=samples.device), torch.empty(1, device=samples.device)
        #     # continue
        targets = target_events
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if is_binary:
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(),
                                          ["accuracy"], is_binary)["accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True, store_embedings = False, path_emb_pkl = "emb.pkl"):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    if store_embedings:
        emb_for_store = []
        target_for_store = []

    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # if step > 2:
        #     break
        EEG = batch[0]
        target = batch[-1]
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(EEG, input_chans=input_chans)
            loss = criterion(output, target)

        if store_embedings:
            target_for_store.append(target.cpu().numpy())
            emb_for_store.append(output.cpu().numpy())

        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if store_embedings:
        with open(path_emb_pkl, 'wb') as handle:
            pickle.dump([emb_for_store, target_for_store], handle)  # , protocol=pickle.HIGHEST_PROTOCOL)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret


@torch.no_grad()
def evaluate_f1_sz_chalenge2025(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True,max_batch_size = 100, XGB_model=None, optimal_threshold=0.5, store_embedings = False, path_emb_pkl = "emb.pkl",add_original=False):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()
    f1_all = []
    sens_all = []
    f_names_all = []
    if add_original:
        pred = []
        true = []
    fs = 200
    if store_embedings:
        signal_for_store = []
        emb_for_store = []
        target_for_store = []

    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
     #   print(batch[1])
        EEG = batch[0]
        if XGB_model:
            EEG_xgb = batch[0]
            EEG_xgb = resample_tensor(EEG_xgb[0, :, :], new_freq=1)

        target = batch[-1].to(device)
        EEG = EEG.float().to(device, non_blocking=True) / 100
        start_pos = torch.tensor(0).to(device)
        end_pos = torch.tensor(batch[0].shape[2]).to(device)
        smple_5_sec = resample_tensor(EEG[0,:, :], new_freq=1)

        if add_original:
            smple_5_sec = EEG
            ev = torch.tensor([0,0,5,target[0].cpu().tolist()]).to(target.device)
            ref = labram_events_to_sz_events(ev.expand(1,4), start_pos / fs, end_pos / fs).to(device)
          #  target[0].expand(1)
        else:
            ref = labram_events_to_sz_events(target[0], start_pos / fs, end_pos / fs).to(device)

        smple_5_sec = rearrange(smple_5_sec, 'B N (A T) -> B N A T', T=200).float().to(device, non_blocking=True)

        chank_size = 2000
        n_chankes = smple_5_sec.shape[0] // chank_size
        all_answer = np.zeros([smple_5_sec.shape[0], 6])

        # outputs = model(smple_5_sec, input_chans = input_chans)
        for i in range(n_chankes + 1):
            answer, emb = model(smple_5_sec[i*chank_size:(i+1)*chank_size,:], input_chans=input_chans)
            hyp = (answer.softmax(dim=1))[:, 0:3].sum(dim=1)

            if store_embedings:
                fname = batch[1]
                istart = i*chank_size
                current_chank = smple_5_sec[i*chank_size:(i+1)*chank_size,:]
                len_chank = current_chank.shape[0]
                signal_for_store.append([fname,istart,len_chank])
                ref_mask = mask_from_events(ref, (smple_5_sec[i*chank_size:(i+1)*chank_size,:].shape[0]))
                target_for_store.append(ref_mask.cpu().numpy())
                emb_for_store.append([answer.cpu().numpy(),emb.cpu().numpy()])

            if XGB_model:
                to_pred = []
                EEG_xgb_chunk =  EEG_xgb[i*chank_size:(i+1)*chank_size]
                for chunk in EEG_xgb_chunk:
                    coefficients = pywt.wavedec2(chunk.numpy(), wavelet='haar', level=4)
                    X = coefficients[0]
                    to_pred.append(X.reshape(126))

                artefact_removing_prob = XGB_model.predict_proba(to_pred)
                artefact_removing_scores= (artefact_removing_prob[:, 3:]).max(1) / ((artefact_removing_prob[:, :3]).max(1) + (artefact_removing_prob[:, 3:]).max(1))
                hyp = torch.from_numpy(artefact_removing_scores < optimal_threshold).to(device) * (hyp)  # if no artefact then artefact_score<th == 1 and no change



            f1, sens = f1_sz_estimation(hyp, ref)

            add_loss_for_f1_sz_estimation = torch.where(
                torch.isnan(f1), torch.tensor(1.0, device=device),  # Штраф за отсутствие детекции
                1 - f1  # Основное значение ошибки
            )
            loss = add_loss_for_f1_sz_estimation.to(device)



            results = {}
            results['f1'] = f1
            results['sensitivity'] = sens
            f1_all.append(f1)
            if add_original:
                output = torch.sigmoid(answer).cpu()
                pred.append(output)
                true.append(target.cpu())

            f_names_all.append(batch[1][0])
            sens_all.append(sens)

            batch_size = EEG.shape[0]
            metric_logger.update(loss=loss.item())
            for key, value in results.items():
                metric_logger.meters[key].update(value, n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            # pred.append((hyp>0.5).float().cpu().numpy().tolist())
            # true.append((target < 3).float().cpu().numpy())  # have to transform from ref to target for every second
       # break
    if store_embedings:
        with open(path_emb_pkl, 'wb') as handle:
            pickle.dump([signal_for_store,emb_for_store,target_for_store], handle) #, protocol=pickle.HIGHEST_PROTOCOL)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    if add_original:
        ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
        return ret

    has_incorrect_files = False

    # Словарь для хранения значений F1 и Sens для корректных объектов
    grouped_f1 = {}
    grouped_sens = {}
    if not add_original:
        for name, f1_value, sens_value in zip(f_names_all, f1_all, sens_all):
            try:
                object_number = int(name.split('-')[1].split('_')[0])
                if object_number not in grouped_f1:
                    grouped_f1[object_number] = []
                    grouped_sens[object_number] = []

                grouped_f1[object_number].append(f1_value)
                grouped_sens[object_number].append(sens_value)

            except (ValueError, IndexError):
                has_incorrect_files = True
                break
    else:
        has_incorrect_files = True

    if has_incorrect_files:
        # Если обнаружены некорректные файлы, усредняем по всем файлам
        overall_average_f1 = sum(f1_all) / len(f1_all)
        overall_average_sens = sum(sens_all) / len(sens_all)
    else:
        # Усредняем значения F1 для каждого объекта
        average_f1_per_object = {obj_num: sum(values) / len(values) for obj_num, values in grouped_f1.items()}
        average_sens_per_object = {obj_num: sum(values) / len(values) for obj_num, values in grouped_sens.items()}

        # Усредняем средние значения по всем объектам
        overall_average_f1 = sum(average_f1_per_object.values()) / len(average_f1_per_object)
        overall_average_sens = sum(average_sens_per_object.values()) / len(average_sens_per_object)


    ret = {}
    ret['loss'] = metric_logger.loss.global_avg
    ret['f1'] = float(overall_average_f1.detach().cpu().numpy())
    ret['sensitivity'] = float(overall_average_sens.detach().cpu().numpy())
    # ret['balanced_accuracy'] = multiclass_metrics_fn(true, pred, metrics=['balanced_accuracy'])

    return ret



@torch.no_grad()
def evaluate_f1_sz_chalenge2025_with_file_splitting_max_batch_size(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True,max_batch_size = 5000):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()
    f1_all = []
    sens_all = []
    f_names_all = []
    fs = 200
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        print(batch[1])
        EEG = batch[0]
        target = batch[-1].to(device)
        EEG = EEG.float().to(device, non_blocking=True) / 100

        batch_size_ = int(batch[0].shape[2]/(max_batch_size*fs))
        for ii in range(batch_size_):
            start_pos = torch.tensor(ii*max_batch_size* fs).to(device)
            end_pos = torch.tensor((ii+1)*max_batch_size* fs).to(device)
            smple_5_sec = resample_tensor(EEG[0,:, start_pos:end_pos], new_freq=1)
            smple_5_sec = rearrange(smple_5_sec, 'B N (A T) -> B N A T', T=200).float().to(device, non_blocking=True)
            ref = labram_events_to_sz_events(target[0], start_pos / fs, end_pos / fs).to(device)
            outputs = model(smple_5_sec, input_chans=input_chans)
            hyp = (outputs.softmax(dim=1))[:, 0:3].sum(dim=1)
            f1, sens = f1_sz_estimation(hyp, ref)

            add_loss_for_f1_sz_estimation = torch.where(
                torch.isnan(f1), torch.tensor(1.0, device=device),  # Штраф за отсутствие детекции
                1 - f1  # Основное значение ошибки
            )
            loss = add_loss_for_f1_sz_estimation.to(device)


            results = {}
            results['f1'] = f1
            results['sensitivity'] = sens
            f1_all.append(f1)
            f_names_all.append(batch[1][0])
            sens_all.append(sens)

            batch_size = EEG.shape[0]
            metric_logger.update(loss=loss.item())
            for key, value in results.items():
                metric_logger.meters[key].update(value, n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    has_incorrect_files = False

    # Словарь для хранения значений F1 и Sens для корректных объектов
    grouped_f1 = {}
    grouped_sens = {}

    for name, f1_value, sens_value in zip(f_names_all, f1_all, sens_all):
        try:
            object_number = int(name.split('-')[1].split('_')[0])
            if object_number not in grouped_f1:
                grouped_f1[object_number] = []
                grouped_sens[object_number] = []

            grouped_f1[object_number].append(f1_value)
            grouped_sens[object_number].append(sens_value)

        except (ValueError, IndexError):
            has_incorrect_files = True
            break

    if has_incorrect_files:
        # Если обнаружены некорректные файлы, усредняем по всем файлам
        overall_average_f1 = sum(f1_all) / len(f1_all)
        overall_average_sens = sum(sens_all) / len(sens_all)
    else:
        # Усредняем значения F1 для каждого объекта
        average_f1_per_object = {obj_num: sum(values) / len(values) for obj_num, values in grouped_f1.items()}
        average_sens_per_object = {obj_num: sum(values) / len(values) for obj_num, values in grouped_sens.items()}

        # Усредняем средние значения по всем объектам
        overall_average_f1 = sum(average_f1_per_object.values()) / len(average_f1_per_object)
        overall_average_sens = sum(average_sens_per_object.values()) / len(average_sens_per_object)


    ret = {}
    ret['loss'] = metric_logger.loss.global_avg
    ret['f1'] = float(overall_average_f1.detach().cpu().numpy())
    ret['sensitivity'] = float(overall_average_sens.detach().cpu().numpy())

    return ret

