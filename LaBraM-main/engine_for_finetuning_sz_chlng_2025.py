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
from pyarrow.compute import random
from timm.utils import ModelEma
import utils
from einops import rearrange

# from ML_solution.infer_model import device
from sz_metrics import f1_sz_estimation, labram_events_to_sz_events, events_from_mask, mask_from_events
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
        end = int(((event_data_torch[0,i,1] + 3) * fs).round())
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


def my_loss(y_pred, y_true, criterion):
    # Получение индексов классов
    indices = (y_pred.argmax(1) < 3).long().view(-1, 1)

    # Применение softmax для нормализации вероятностей
    y_pred_softmax = softmax(y_pred, dim=1)

    # Сборка нужных значений из нормализованных вероятностей
    y_pred_filtered = torch.gather(y_pred_softmax, 1, indices).squeeze(1)

    # Преобразование к типу float
    y_pred_filtered = y_pred_filtered.to(dtype=torch.float)

    # Логарифмические вероятности для y_pred
    log_probs = log_softmax(y_pred_filtered.unsqueeze(1), dim=-1)

    # Целевые классы
    target_indices = y_true.view(-1, 1).float()

    # Потеря
    loss = criterion(log_probs, target_indices)

    return loss

def my_loss11(y_pred, y_true, criterion):
    """ for debug, to use crossentrapy instead of MSELoss"""
    indices = (y_pred.argmax(1) < 3).long().view(-1, 1)
    y_pred_softmax = softmax(y_pred, dim=1)
    y_pred_filtered = gather(y_pred_softmax, 1, indices).squeeze(1)
    y_pred_filtered = y_pred_filtered.to(dtype=torch.float)

    log_probs = log_softmax(y_pred_filtered.unsqueeze(1), dim=-1)
    target_indices = torch.argmax(y_true, dim=1).unsqueeze(1)

    # y_true_onehot = one_hot_encoding(y_true)
    # y_pred_onehot = one_hot_encoding(y_pred_filtered)

    # logits = torch.log(y_pred_onehot + 1e-10)

    return criterion(log_probs, target_indices)


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
    outputs = model(smple_5_sec, ch_names)
    prob = F.softmax(outputs, dim=1)
    # Вместо дискретного argmax используем среднее по вероятностям для формирования soft-выхода
    hyp = 1 - prob[:, :3].mean(dim=1)
    hyp_event = events_from_mask(hyp, fs)
    f1, sens = f1_sz_estimation(hyp, ref, start_pos / fs, end_pos / fs, fs=1)
    add_loss_for_f1_sz_estimation = torch.where(
        torch.isnan(f1),
        torch.tensor(1.0, device=samples.device),
        1 - f1
    )
    loss = add_loss_for_f1_sz_estimation.to(samples.device)
    ref_mask = mask_from_events(ref, smple_5_sec.shape[0], fs)
    return loss, outputs, ref_mask


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

    for data_iter_step, (samples, fname, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print("start file:",fname)   # debug
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

        if loss_scaler is None:
            samples = samples.half()
            loss, output, target_events = train_class_batch(
                model, samples, targets, criterion, input_chans, max_batch_size)
        else:
            with torch.cuda.amp.autocast():
                loss, output, target_events = train_class_batch(
                    model, samples, targets, criterion, input_chans, max_batch_size)
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
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True):
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
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
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
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret