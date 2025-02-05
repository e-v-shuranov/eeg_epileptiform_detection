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
from timm.utils import ModelEma
import utils
from einops import rearrange
from sz_metrics import f1_sz_estimation


def resample_tensor(tensor, new_freq=10):
    # Исходная частота дискретизации
    old_freq = 200

    # Размер исходного тензора
    channels , time_len  = tensor.shape

    # Время одной выборки в секундах
    dt_old = 1 / old_freq
    dt_new = 1 / new_freq

    # Количество временных шагов в новой временной шкале
    new_time_len = int(time_len * dt_old // dt_new)

    # Новый тензор для хранения результата
    new_tensor = torch.zeros((new_time_len, channels, 5*old_freq))

    # Границы интервала для каждого нового временного шага
    window_start = -int(2 * old_freq)
    window_end = int(3 * old_freq)

    for i in range(new_time_len):
        t_center = int(i * dt_new / dt_old)

        start_idx = max(window_start + t_center, 0)
        end_idx = min(window_end + t_center, time_len)

        if start_idx == 0:
            # Если начало окна выходит за пределы начала тензора,
            # используем padding
            pad_left = abs(int(5 * old_freq) - end_idx)
            padded_slice = torch.cat([tensor[:,:pad_left], tensor[:,start_idx:end_idx]], dim=1)
        elif end_idx == time_len:
            # Если конец окна выходит за пределы конца тензора,
            # используем padding
            pad_right = abs(2*time_len - int(5 * old_freq) - start_idx )
            padded_slice = torch.cat([tensor[:,start_idx:end_idx], tensor[:,pad_right:]], dim=1)
        else:
            # Если окно полностью внутри границ тензора
            padded_slice = tensor[:,start_idx:end_idx]

        new_tensor[i] = padded_slice

    return new_tensor


def get_events_based_data(signals_torch, event_data_torch):
    # Преобразование входных данных в тензоры PyTorch
    # signals_torch = torch.from_numpy(signals).float()
    # times_torch = torch.from_numpy(times).float()
    # event_data_torch = torch.from_numpy(event_data).long()

    # Параметры
    fs = 10.0
    num_channels, num_points = signals_torch.shape
    # num_events = event_data_torch.size(0)

    # Вычисление индексов для срезов
    feature_index_start = event_data_torch[:, 1].sub_(2 * fs).mul_(-1).add_(num_points)
    feature_index_end = event_data_torch[:, 2].add_(2 * fs).mul_(-1).add_(num_points)

    # # Создание пустого тензора для результатов
    # features = torch.empty(num_events, num_channels, int(fs) * 5)
    #
    # # Заполнение тензоров
    # for i in range(num_events):
    #     features[i] = signals_torch[:, feature_index_start[i]:feature_index_end[i]]

    # Индексы событий
    feature_indexes = torch.stack([feature_index_start, feature_index_end], dim=-1)

    # Метки и номера каналов
    # offending_channel = event_data_torch[:, 0].unsqueeze(dim=1)
    labels = event_data_torch[:, 3].unsqueeze(dim=1)

    return feature_indexes, labels

def train_class_batch(model, samples, target, criterion, ch_names, max_batch_size):
    old_fs = 200  # in samples
    new_fs = 10   # in outputs

    n_samples = int(samples.shape[2] /old_fs* new_fs)
    n_batches = int(n_samples / max_batch_size + 1)
    outputs = torch.zeros((n_samples, 6)).to(samples.device, non_blocking=True)
    for i in range(n_batches):
        start_pos = int(i*max_batch_size/new_fs*old_fs)
        end_pos = int((i+1)*max_batch_size/new_fs*old_fs)
        smple_5_sec = resample_tensor(samples[0,:,start_pos:end_pos], new_freq=10)
        # smple_5_sec = resample_tensor(samples[i*max_batch_size:(i+1)*max_batch_size], new_freq=10)
        smple_5_sec = rearrange(smple_5_sec, 'B N (A T) -> B N A T', T=200).float().to(samples.device, non_blocking=True)
        outputs[i*max_batch_size:(i+1)*max_batch_size] = model(smple_5_sec, ch_names)
    outputs_indexes_of_events, targets = get_events_based_data(outputs, target)
    loss = criterion(outputs[outputs_indexes_of_events], targets)

    # ref = (target < 3)
    # hyp = (outputs.argmax(1) < 3)
    # add_loss_for_f1_sz_estimation=1 - f1_sz_estimation(hyp,ref)
    add_loss_for_f1_sz_estimation = 0
    alpha = 1
    return loss + alpha * add_loss_for_f1_sz_estimation, outputs


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

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans, max_batch_size)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans, max_batch_size)

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