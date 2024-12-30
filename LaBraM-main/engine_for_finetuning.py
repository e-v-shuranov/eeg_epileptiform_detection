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
import pywt

def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
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
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

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
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
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
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True, is_mbt = False, from_multiclass_to_binary=False):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    correct = 0
    count_all = 0
    use_thresholds_for_artefacts = True
    threshold_for_artefacts = 0
    threshold_for_epilepsy = -5
    art = 0
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

        if is_mbt:
            if use_thresholds_for_artefacts:
                output_artefacts = (output[:,3:6].max(dim=1)[0]>threshold_for_artefacts)
                output = (~output_artefacts)*((output.max(dim=1)[0]>threshold_for_epilepsy) * output.argmax(dim=1)<3).float()
            else:
                output = (output.argmax(dim=1)<3).float()
            target =  (target>0).float()
            loss = criterion(output, target) # 0,1,2 - sizors  3,4,5 - artefacts
            output = output.int().cpu()
            target = target.int()
            correct += (output == target.cpu()).int().sum()
            count_all += output.size(0)
            if use_thresholds_for_artefacts:
                art += (output_artefacts.cpu()).int().sum()
        else:
            loss = criterion(output, target)
            if is_binary:
                output = torch.sigmoid(output).cpu()
            else:
                output = output.cpu()

        if (not is_mbt) and from_multiclass_to_binary:
            if 'f1_weighted' in metrics:
                metrics.remove('f1_weighted')
            if use_thresholds_for_artefacts:
                output_artefacts = (output[:,3:6].max(dim=1)[0]>threshold_for_artefacts)
                output = (~output_artefacts)*((output.max(dim=1)[0]>threshold_for_epilepsy) * output.argmax(dim=1)<3).float()
            else:
                output = (output.argmax(dim=1)<3).float()
            target = (target < 3).float()   # 2 classes only

        target = target.cpu()


        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, (is_binary or is_mbt or from_multiclass_to_binary))
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, (is_binary or is_mbt or from_multiclass_to_binary), 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    if is_mbt:
        if use_thresholds_for_artefacts:
            print("mbt:  correct: ", correct, "All: ", count_all, "acc: ", correct/count_all, "art: ", art)
        else:
            print("mbt:  correct: ", correct, "All: ", count_all, "acc: ", correct/count_all)
    return ret


import numpy as np
import scipy
import matplotlib.pyplot as plt
def get_amplitude(x):
    spec = scipy.fft.fft(x)

    ampl = np.absolute(spec)
    ff = np.arange(0,100, (1./(1./200*ampl.size)))

    plt.plot(ff[5:int(ampl.size/2)],ampl[5:int(ampl.size/2)], color="blue")
    plt.show()

    return




import csv

@torch.no_grad()
def evaluate_for_mbt_binary_scenario(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'],
                                     is_binary=True, is_mbt=False, use_thresholds_for_artefacts=True,
                                     threshold_for_artefacts=2.11, threshold_for_epilepsy=1,
                                     path_output = "log_output.csv", metrics_for_interval_label=True, XGB_model=None):

    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    correct = 0
    count_all = 0

    time_index = []
    out_label = []
    art = 0
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        EEG_xgb = batch[0]

        target = batch[-1]
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        # for i in range(batch[0].size()[0]):
        #     for ch in range(batch[0].size()[1]):
        #         get_amplitude(batch[0][i][ch])
        # compute output
        with torch.cuda.amp.autocast():
            output = model(EEG, input_chans=input_chans)

        # torch.cat((EEG, EEG), dim=2)

        if use_thresholds_for_artefacts:
            output_artefacts = (output[:, 3:6].max(dim=1)[0] > threshold_for_artefacts)
            output = (~output_artefacts) * (
                        (output.max(dim=1)[0] > threshold_for_epilepsy) * (output.argmax(dim=1) < 3)).float()
        else:
            output = (output.argmax(dim=1) < 3).float()

        time_index.extend(target.cpu().numpy())
        out_label.extend(output.cpu().numpy())

        if is_mbt:
            target = (target > -0.01).float()
            output = output.int().cpu()   # 0,1,2 - sizors  3,4,5 - artefacts
            target = target.int()
            correct += (output == target.cpu()).int().sum()
            count_all += output.size(0)
            if use_thresholds_for_artefacts:
                art += (output_artefacts.cpu()).int().sum()

        else:  # TUEV
            target = (target < 3).float()  # 2 classes only
        loss = criterion(output.cpu().float(), target.cpu().float())

        if XGB_model:
            to_pred = []
            for file in EEG_xgb:
                coefficients = pywt.dwt(file.numpy(), 'haar')  # Perform discrete Haar wavelet transform
                X = coefficients[0]
                to_pred.append(X.reshape(4000))

            xgb_answer = XGB_model.predict(to_pred)
            binary_ans = []
            for ans in xgb_answer:
                if ans > 3:
                    binary_ans.append(0)
                else:
                    binary_ans.append(1)

            output = torch.tensor(binary_ans).float()

        else:
            output = output.cpu()
        target = target.cpu()

        # if 1 in target:
        #     print("target: ", target)
        # if 1 in output:
        #     print("output: ", output)

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics,
                                    (is_binary or is_mbt))

        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    #store output
    with open(path_output, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(time_index)
        write.writerow(out_label)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    if metrics_for_interval_label:  # metrics estimation for dataset with labels for long intervals. if at least 1 time on interval label detected - correct.
        True_negative_count = 0
        True_positive_count = 0
        False_negative_count = 0
        False_positive_count = 0

        i_true_begin = -1
        i_true_end = -1
        for i_true in range(1,len(true)):
            if (true[i_true] == 1 and true[i_true-1] == 0) or (i_true==len(true)-1):  # end of 0 interval
                i_true_begin = i_true                                                 # store begin of 1 interval
                is_label_on_0_interval = False
                for i in range(i_true_end+1, i_true_begin):
                    if pred[i] == 1:
                        False_positive_count += 1
                        is_label_on_0_interval = True
                        # break                                                       # lets count all False positive
                if not is_label_on_0_interval:
                    True_negative_count += 1

            if (true[i_true] == 0 and true[i_true-1] == 1) or (i_true_begin>0 and i_true_begin<len(true)-1 and i_true==len(true)-1):  # end of 1 interval
                i_true_end = i_true-1                                                                    # store begin of 0 interval
                is_label_on_1_interval = False
                for i in range(i_true_begin, i_true_end):
                    if pred[i] == 1:
                        True_positive_count += 1
                        is_label_on_1_interval = True
                        break
                if not is_label_on_1_interval:
                    False_negative_count += 1
        print("TN: ",True_negative_count, "TP: ",True_positive_count, "FN: ",False_negative_count, "FP: ",False_positive_count)
        return True_negative_count, True_positive_count, False_negative_count, False_positive_count



    ret = utils.get_metrics(pred, true, metrics, (is_binary or is_mbt), 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    if is_mbt:
        if use_thresholds_for_artefacts:
            print("mbt:  correct: ", correct, "All: ", count_all, "acc: ", correct / count_all, "art: ", art)
        else:
            print("mbt:  correct: ", correct, "All: ", count_all, "acc: ", correct / count_all)
    return ret