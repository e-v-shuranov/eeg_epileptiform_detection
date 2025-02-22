# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from run_class_finetuning import *
# from engine_for_finetuning import train_one_epoch, evaluate
from engine_for_finetuning_sz_chlng_2025 import train_one_epoch_sz_chlng_2025, evaluate, evaluate_f1_sz_chalenge2025, evaluate_f1_sz_chalenge2025_with_file_splitting_max_batch_size

from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import prepare_TUEV_dataset
import utils
from scipy import interpolate
import modeling_finetune

from fusion_model_train import Tfusion_clf

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


# class TUEVLoader(torch.utils.data.Dataset):
#     def __init__(self, root, files, sampling_rate=200):
#         self.root = root
#         sort_nicely(files)
#         self.files = files
#         self.default_rate = 200
#         self.sampling_rate = sampling_rate
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, index):
#         sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
#         X = sample["signal"]
#         if self.sampling_rate != self.default_rate:
#             X = resample(X, 5 * self.sampling_rate, axis=-1)
#         Y = int(sample["label"][0] - 1)
#         # Y = index
#         X = torch.FloatTensor(X)
#         return X, Y


# def prepare_TUEV_dataset(root):
#     # set random seed
#     seed = 4523
#     np.random.seed(seed)
#
#     train_path = os.path.join(root, "train")
#     eval_path = os.path.join(root, "eval")
#     test_path = os.path.join(root, "test")
#
#     train_files = os.listdir(train_path)
#     val_files = os.listdir(eval_path)
#     test_files = os.listdir(test_path)
#
#     # prepare training and test data loader
#     train_dataset = TUEVLoader(train_path, train_files)
#     test_dataset = TUEVLoader(test_path, test_files)
#     val_dataset = TUEVLoader(eval_path, val_files)
#
#     print(len(train_files), len(val_files), len(test_files))
#     return train_dataset, test_dataset, val_dataset


class TUEVLoader_sz_chalenge_2025_full_files(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        sort_nicely(files)
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = sample["event"]
        # Y = index
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        return X, self.files[index], Y


class All_Loader_sz_chalenge_2025_full_files(torch.utils.data.Dataset):
    def __init__(self, files):
        sort_nicely(files)
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        # sample = pickle.load(open(os.path.join(self.root, "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_to_labram_pkl/all/sub-12_ses-01_task-szMonitoring_run-02_events.pkl"), "rb"))
        X = sample["signal"]
        Y = sample["event"]
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        return X, self.files[index], Y

def prepare_TUEV_dataset_sz_chalenge_2025_full_files(root):
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_path = os.path.join(root, "train")
    eval_path = os.path.join(root, "eval")
    test_path = os.path.join(root, "test")

    train_files = os.listdir(train_path)
    val_files = os.listdir(eval_path)
    test_files = os.listdir(test_path)

    # prepare training and test data loader
    train_dataset = TUEVLoader_sz_chalenge_2025_full_files(train_path, train_files)
    test_dataset = TUEVLoader_sz_chalenge_2025_full_files(test_path, test_files)
    val_dataset = TUEVLoader_sz_chalenge_2025_full_files(eval_path, val_files)

    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def get_full_paths(directory):
    file_list = os.listdir(directory)
    full_paths = [os.path.join(directory, f) for f in file_list]
    return full_paths

def prepare_All_dataset_sz_chalenge_2025_full_files(All_paths):
    # set random seed
    seed = 4523
    np.random.seed(seed)
    Siena_path, TUSZ_path, TUEV_path = All_paths

    TUEV_train_path = os.path.join(TUEV_path, "train")
    TUEV_eval_path = os.path.join(TUEV_path, "eval")
    TUEV_test_path = os.path.join(TUEV_path, "test")

    TUEV_train_files = get_full_paths(TUEV_train_path)
    TUEV_val_files = get_full_paths(TUEV_eval_path)[:2]
    TUEV_test_files = get_full_paths(TUEV_test_path)[:1500]

    Siena_train_path = os.path.join(Siena_path, "train")
    Siena_eval_path = os.path.join(Siena_path, "eval")

    Siena_train_files = get_full_paths(Siena_train_path)
    Siena_val_files = get_full_paths(Siena_eval_path)[:1500]

    TUSZ_train_path = os.path.join(TUSZ_path, "train")
    TUSZ_eval_path = os.path.join(TUSZ_path, "eval")

    TUSZ_train_files = get_full_paths(TUSZ_train_path)
    TUSZ_val_files = get_full_paths(TUSZ_eval_path)[:1500]


    # prepare training and test data loader
    train_dataset = All_Loader_sz_chalenge_2025_full_files(Siena_train_files+TUSZ_train_files+TUEV_train_files)
    test_dataset = All_Loader_sz_chalenge_2025_full_files(TUEV_test_files)
    val_dataset = All_Loader_sz_chalenge_2025_full_files(Siena_val_files+TUSZ_val_files+TUEV_val_files)

    print(len(Siena_train_files+TUSZ_train_files+TUEV_train_files), len(Siena_val_files+TUSZ_val_files+TUEV_val_files), len(TUEV_test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_Siena_dataset_sz_chalenge_2025_full_files(root):
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_path = os.path.join(root, "train")
    eval_path = os.path.join(root, "eval")
    test_path = os.path.join(root, "eval")   # just copy to fit other datasets structure

    train_files = os.listdir(train_path)
    val_files = os.listdir(eval_path)
    test_files = os.listdir(test_path)

    # prepare training and test data loader
    train_dataset = TUEVLoader_sz_chalenge_2025_full_files(train_path, train_files)
    test_dataset = TUEVLoader_sz_chalenge_2025_full_files(test_path, test_files)
    val_dataset = TUEVLoader_sz_chalenge_2025_full_files(eval_path, val_files)

    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def get_dataset(args):
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("path/to/TUAB")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif True:  # args.dataset == 'TUEV':   # we have only TUEV for finetune  labram classes
        train_dataset, test_dataset, val_dataset = prepare_TUEV_dataset("/media/public/Datasets/TUEV/tuev/edf/processed_sz_chalenge_2025_montage")
        ch_names_after_convert_sz_chalenge_2025_montage = ['FP1-Avg', 'F3-Avg', 'C3-Avg', 'P3-Avg',
                                                           'O1-Avg', 'F7-Avg', 'T3-Avg', 'T5-Avg',
                                                           'FZ-Avg', 'CZ-Avg', 'PZ-Avg', 'FP2-Avg',
                                                           'F4-Avg', 'C4-Avg', 'P4-Avg', 'O2-Avg',
                                                           'F8-Avg', 'T4-Avg', 'T6-Avg']
        # Let's replace ch_names_after_convert_sz_chalenge_2025_montage to standard_1020_subset names and fintune for "-Avg"
        standard_1020_subset = ['FP1', 'F3', 'C3', 'P3',
                               'O1', 'F7', 'T3', 'T5',
                               'FZ', 'CZ', 'PZ', 'FP2',
                               'F4', 'C4', 'P4', 'O2',
                               'F8', 'T4', 'T6']
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    return train_dataset, test_dataset, val_dataset, standard_1020_subset, metrics


def get_dataset_sz_chalenge_2025_full_files(args):
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("path/to/TUAB")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.dataset == 'All':
        # Siena_test_path = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_to_labram_pkl/test1"
        TUEV_path = "/media/public/Datasets/TUEV/tuev/edf/processed_sz_chalenge_2025_full_file_sz_only"
        Siena_path = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_to_labram_pkl/splited"
        TUSZ_path = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess_pkl/splited"

        train_dataset, test_dataset, val_dataset = prepare_All_dataset_sz_chalenge_2025_full_files([Siena_path, TUSZ_path, TUEV_path])

        ch_names_after_convert_sz_chalenge_2025_montage = ['FP1-Avg', 'F3-Avg', 'C3-Avg', 'P3-Avg',
                                                           'O1-Avg', 'F7-Avg', 'T3-Avg', 'T5-Avg',
                                                           'FZ-Avg', 'CZ-Avg', 'PZ-Avg', 'FP2-Avg',
                                                           'F4-Avg', 'C4-Avg', 'P4-Avg', 'O2-Avg',
                                                           'F8-Avg', 'T4-Avg', 'T6-Avg']
        # Let's replace ch_names_after_convert_sz_chalenge_2025_montage to standard_1020_subset names and fintune for "-Avg"
        standard_1020_subset = ['FP1', 'F3', 'C3', 'P3',
                               'O1', 'F7', 'T3', 'T5',
                               'FZ', 'CZ', 'PZ', 'FP2',
                               'F4', 'C4', 'P4', 'O2',
                               'F8', 'T4', 'T6']
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        return train_dataset, test_dataset, val_dataset, standard_1020_subset, metrics
    elif args.dataset == 'Siena':
        train_dataset, test_dataset, val_dataset = prepare_Siena_dataset_sz_chalenge_2025_full_files("/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_to_labram_pkl/test1")
        ch_names_after_convert_sz_chalenge_2025_montage = ['FP1-Avg', 'F3-Avg', 'C3-Avg', 'P3-Avg',
                                                           'O1-Avg', 'F7-Avg', 'T3-Avg', 'T5-Avg',
                                                           'FZ-Avg', 'CZ-Avg', 'PZ-Avg', 'FP2-Avg',
                                                           'F4-Avg', 'C4-Avg', 'P4-Avg', 'O2-Avg',
                                                           'F8-Avg', 'T4-Avg', 'T6-Avg']
        # Let's replace ch_names_after_convert_sz_chalenge_2025_montage to standard_1020_subset names and fintune for "-Avg"
        standard_1020_subset = ['FP1', 'F3', 'C3', 'P3',
                               'O1', 'F7', 'T3', 'T5',
                               'FZ', 'CZ', 'PZ', 'FP2',
                               'F4', 'C4', 'P4', 'O2',
                               'F8', 'T4', 'T6']
        args.nb_classes = 6  # use same logic like TUEV, but labels only 2 and 3 exists
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    return train_dataset, test_dataset, val_dataset, standard_1020_subset, metrics


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)
    dataset_train_sz_chalenge_2025_full_files, dataset_test_sz_chalenge_2025_full_files, dataset_val_sz_chalenge_2025_full_files, ch_names_sz_chalenge_2025_full_files, metrics_sz_chalenge_2025_full_files = get_dataset_sz_chalenge_2025_full_files(args)

    dataset_test = [dataset_test,dataset_test_sz_chalenge_2025_full_files,dataset_val_sz_chalenge_2025_full_files]
    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train_sz_chalenge_2025_full_files, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train_sz_chalenge_2025_full_files)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train_sz_chalenge_2025_full_files, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train_sz_chalenge_2025_full_files) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train_sz_chalenge_2025_full_files))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if False: #args.eval:
        # balanced_accuracy = []
        # accuracy = []
        # for data_loader in data_loader_test:
        #     test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
        #                           is_binary=(args.nb_classes == 1))
        #     accuracy.append(test_stats['accuracy'])
        #     balanced_accuracy.append(test_stats['balanced_accuracy'])
        # print(
        #     f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")

        # test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
        #                       is_binary=(args.nb_classes == 1))
        # print(f"======Accuracy: on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%", "ALL tests: ",
        #       test_stats)
        test_stats = evaluate_f1_sz_chalenge2025(data_loader_test[1], model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                              is_binary=(args.nb_classes == 1))

        print(f"test_stats on the {len(dataset_test[1])} test EEG: ",test_stats)

        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    max_f1_val_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch_sz_chlng_2025(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names, is_binary=args.nb_classes == 1, max_batch_size = args.max_batch_size
        )

        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            test_stats = evaluate(data_loader_test[0], model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            print(f"labramtest: Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")
            f1_test_stats = evaluate_f1_sz_chalenge2025_with_file_splitting_max_batch_size(data_loader_test[1], model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1, max_batch_size = args.max_batch_size)
            print(f"f1_sz_2025_test: f1 of the network on the {len(dataset_test_sz_chalenge_2025_full_files)} test EEG: {f1_test_stats['f1']:.2f}%")
            f1_val__stats = evaluate_f1_sz_chalenge2025_with_file_splitting_max_batch_size(data_loader_test[2], model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1, max_batch_size = args.max_batch_size)
            print(f"f1_sz_2025_val: f1 of the network on the {len(dataset_val_sz_chalenge_2025_full_files)} test EEG: {f1_val__stats['f1']:.2f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                max_accuracy_test = test_stats["accuracy"]

            if max_f1_val_accuracy < f1_val__stats["f1"]:
                max_f1_val_accuracy = f1_val__stats["f1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best_f1_val_2025", model_ema=model_ema)
                max_f1_accuracy_test = f1_test_stats["f1"]

            print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
            print(f'f1_sz_2025: Max accuracy val: {max_f1_val_accuracy:.2f}%, max accuracy test: {max_f1_accuracy_test:.2f}%')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)

                for key, value in f1_val__stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="f1_val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="f1_val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="f1_val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="f1_val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="f1_val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="f1_val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="f1_val", step=epoch)

                    for key, value in f1_test_stats.items():
                        if key == 'accuracy':
                            log_writer.update(accuracy=value, head="f1_test", step=epoch)
                        elif key == 'balanced_accuracy':
                            log_writer.update(balanced_accuracy=value, head="f1_test", step=epoch)
                        elif key == 'f1_weighted':
                            log_writer.update(f1_weighted=value, head="f1_test", step=epoch)
                        elif key == 'pr_auc':
                            log_writer.update(pr_auc=value, head="f1_test", step=epoch)
                        elif key == 'roc_auc':
                            log_writer.update(roc_auc=value, head="f1_test", step=epoch)
                        elif key == 'cohen_kappa':
                            log_writer.update(cohen_kappa=value, head="f1_test", step=epoch)
                        elif key == 'loss':
                            log_writer.update(loss=value, head="f1_test", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         **{f'f1_val_{k}': v for k, v in f1_val__stats.items()},
                         **{f'f1_test_{k}': v for k, v in f1_test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)