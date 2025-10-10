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

from engine_for_finetuning_multidata import train_one_epoch, evaluate
from utils_multidata import NativeScalerWithGradNormCount as NativeScaler
import utils_multidata
from scipy import interpolate
import modeling_finetune

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--check_dataset', action='store_true', default=False,
                        help='Run or Do not run sanity_check_splits ')


    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--datasets_dir', default='',
                        help='path to dataset for finetune and test in multidata (multi downstream tasks) experiments')
    parser.add_argument('--fixed_chkpt', default='',
                        help='path for fixed checkpoint for evaluation or continue')

    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Positive-class weight for BCEWithLogitsLoss when using binary datasets')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV | CHB-MIT')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )

    return model


def _normalize_chb_mit_channels(raw_channels):
    """Convert CHB-MIT channel labels to the standard_1020 bipolar subset."""

    alias_map = {
        'F7-T3': 'F7-T7',
        'T3-T5': 'T7-P7',
        'T5-O1': 'P7-O1',
        'T6-O2': 'P8-O2',
    }

    normalized_channels = []
    for channel in raw_channels:
        normalized = channel.strip().upper().replace(' ', '')
        normalized = alias_map.get(normalized, normalized)
        if normalized not in utils_multidata.standard_1020:
            raise ValueError(
                f"Channel '{channel}' normalized to '{normalized}' is not present in standard_1020"
            )
        normalized_channels.append(normalized)

    return normalized_channels


def get_dataset(args):

    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils_multidata.prepare_TUAB_dataset("path/to/TUAB")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils_multidata.prepare_TUEV_dataset("/media/public/Datasets/labram_data/TUEV/processed")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif args.dataset == 'Mumtaz':
        load_dataset = mumtaz_dataset.LoadDataset(args)
        train_dataset, val_dataset, test_dataset = load_dataset.get_data_loader()
        ch_names = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                     'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                     'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE']

        ch_names = [x.upper() for x in ch_names]
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.dataset == 'ISRUC':
        load_dataset = isruc_dataset.LoadDataset(args)
        train_dataset, val_dataset, test_dataset = load_dataset.get_data_loader()
        #F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1
        ch_names = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']

        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif args.dataset == 'FACED':
        load_dataset = faced_dataset.LoadDataset(args)
        train_dataset, val_dataset, test_dataset = load_dataset.get_data_loader()
        #F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1
       # ch_names = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        ch_names_original = [
            "Cz",
            "CP1",
            "C4",
            "Pz",
            "O1",
            "Fp1",
            "CP6",
            "O2",
            "CP2",
            "P8",
            "C3",
            "P7",
            "Fp2",
            "F8",
            "CP5",
            "F7",
            "HEOL",
            "FC5",
            "HEOR",
            "P4",
            "T7",
            "F3",
            "T8",
            "FC2",
            "PO4",
            "FC6",
            "P3",
            "PO3",
            "Oz",
            "F4",
            "FC1",
            "Fz"
        ]
        ch_names_original = ch_names_original[:16] + ch_names_original[17:18] + ch_names_original[19:]
        ch_names = [word.upper() for word in ch_names_original]
        args.nb_classes = 9
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif args.dataset == 'PHYSIO':
        load_dataset = physio_dataset.LoadDataset(args)
        train_dataset, val_dataset, test_dataset = load_dataset.get_data_loader()
        selected_channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..',
                             'C2..',
                             'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.',
                             'Fp2.',
                             'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..',
                             'F4..',
                             'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..',
                             'P5..',
                             'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.',
                             'Po8.',
                             'O1..', 'Oz..', 'O2..', 'Iz..']
        new_list = [s.replace(".", "") for s in selected_channels]
        ch_names = [word.upper() for word in new_list]


        args.nb_classes = 4
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif args.dataset == 'CHB-MIT':
        load_dataset = chb_dataset.LoadDataset(args)
        train_dataset, val_dataset, test_dataset = load_dataset.get_data_loader()
        ch_names_original = [
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
        ]
        ch_names_original = ch_names_original[:8]
        # Align to the CHB-MIT bipolar subset included at the end of standard_1020.
        ch_names = _normalize_chb_mit_channels(ch_names_original)
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    return train_dataset, test_dataset, val_dataset, ch_names, metrics

# --- Dataset sanity check: NaN/Inf scan + class balance ----------------------
from collections import Counter, defaultdict
import numpy as np
import torch

def _extract_xy(sample):
    """
    Универсально достаёт (X, y) из sample для разных датасетов.
    Возвращает: X (np.ndarray|Tensor), y (int|None), meta (исходный sample)
    """
    X = None
    y = None

    if isinstance(sample, (list, tuple)):
        # X — первый тензор/массив с ndim >= 2
        for el in sample:
            if X is None and isinstance(el, (np.ndarray, torch.Tensor)) and getattr(el, "ndim", 0) >= 2:
                X = el
        # y — последнее скалярное/1-элементное значение
        for el in reversed(sample):
            if isinstance(el, (int, np.integer)):
                y = int(el)
                break
            if isinstance(el, (np.ndarray, torch.Tensor)) and np.prod(getattr(el, "shape", ())) == 1:
                y = int(el.item())
                break
        return X, y, sample
    else:
        # sample уже (X,?) — используем как есть
        X = sample
        return X, y, sample


def _to_tensor_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        # непредвидимый тип — перевести не можем
        return None


def sanity_check_splits(
    datasets_dict,
    nb_classes=None,
    max_items_per_split=None,  # можно ограничить для быстрого прогона; None = все
    verbose=True,
):
    """
    datasets_dict: {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    nb_classes: если известно (args.nb_classes), проверим попадание меток в диапазон [0, nb_classes-1]
    """
    summary = {}

    for split_name, ds in datasets_dict.items():
        if ds is None:
            summary[split_name] = {"size": 0, "nan": 0, "inf": 0, "bad_samples": [], "class_counts": {}}
            continue

        n = len(ds)
        limit = n if max_items_per_split is None else min(n, max_items_per_split)

        nan_total = 0
        inf_total = 0
        bad_samples = []  # индексы сэмплов, в которых встретились NaN/Inf
        class_counter = Counter()
        out_of_range_labels = defaultdict(int)

        for idx in range(limit):
            try:
                sample = ds[idx]
            except Exception as e:
                # если не удалось прочитать — считаем как «битый»
                bad_samples.append((idx, f"read_error: {e}"))
                continue

            X, y, _meta = _extract_xy(sample)
            xt = _to_tensor_float(X)
            if xt is None:
                bad_samples.append((idx, "unsupported_X_type"))
                continue

            # считаем NaN/Inf на уровне одного сэмпла
            n_nan = torch.isnan(xt).sum().item()
            n_inf = torch.isinf(xt).sum().item()
            if n_nan or n_inf:
                bad_samples.append((idx, f"nan={int(n_nan)}, inf={int(n_inf)}"))

            nan_total += int(n_nan)
            inf_total += int(n_inf)

            # учтём класс, если он есть
            if y is not None:
                class_counter[int(y)] += 1
                if nb_classes is not None and (y < 0 or y >= nb_classes):
                    out_of_range_labels[int(y)] += 1

        summary[split_name] = {
            "size": n,
            "checked": limit,
            "nan": nan_total,
            "inf": inf_total,
            "bad_samples": bad_samples,
            "class_counts": dict(class_counter),
            "labels_out_of_range": dict(out_of_range_labels),
        }

    if verbose:
        print("\n==== Dataset sanity check ====")
        for split_name, stats in summary.items():
            print(f"\n[{split_name}] size={stats['size']} (checked {stats['checked']})")
            print(f"  NaN total: {stats['nan']}, Inf total: {stats['inf']}")
            if stats["labels_out_of_range"]:
                print(f"  ⚠ labels out of range: {stats['labels_out_of_range']}")
            # баланс классов
            if stats["class_counts"]:
                total_labeled = sum(stats["class_counts"].values())
                print(f"  Class balance (count / share):")
                for cls, cnt in sorted(stats["class_counts"].items()):
                    share = 100.0 * cnt / max(1, total_labeled)
                    print(f"    class {cls}: {cnt} ({share:.2f}%)")
            # для отладки покажем первые 10 проблемных индексов
            if stats["bad_samples"]:
                preview = ", ".join([f"{i}:{msg}" for i, msg in stats["bad_samples"][:10]])
                print(f"  Bad samples: {len(stats['bad_samples'])}  e.g. {preview}")
            else:
                print("  Bad samples: 0")

    return summary
# ---------------------------------------------------------------------------



def main(args, ds_init):
    utils_multidata.init_distributed_mode(args)

    if ds_init is not None:
        utils_multidata.create_ds_config(args)

    print(args)

    # device = torch.device(args.device)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = args.seed + utils_multidata.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    if args.check_dataset and utils_multidata.get_rank() == 0:  # Проверка датасетов. сейчас для FACED все отлично
        _ = sanity_check_splits(
            {
                "train": dataset_train,
                "val": dataset_val,
                "test": dataset_test,
            },
            nb_classes=getattr(args, "nb_classes", None),
            max_items_per_split=None,  # можно поставить, например, 200 для быстрого прогона
            verbose=True,
        )


    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils_multidata.get_world_size()
        global_rank = utils_multidata.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
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
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils_multidata.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # def _worker_init_fn(_):
    #     # на случай, если захотите что-то явно сбрасывать в датасете
    #     pass
    pf = None if opts.num_workers == 0 else 2  #opts.prefetch_factor
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=False,  # Критично: не держим воркеров «липкими»
        prefetch_factor=pf,
        # worker_init_fn=_worker_init_fn
    )


    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=False,  # Критично: не держим воркеров «липкими»
            prefetch_factor=pf,
            # worker_init_fn=_worker_init_fn
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                persistent_workers = False,  # Критично: не держим воркеров «липкими»
                prefetch_factor = pf,
                # worker_init_fn = _worker_init_fn
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                persistent_workers = False,  # Критично: не держим воркеров «липкими»
                prefetch_factor = pf,
                # worker_init_fn = _worker_init_fn
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

        utils_multidata.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

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

    total_batch_size = args.batch_size * args.update_freq * utils_multidata.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
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
    lr_schedule_values = utils_multidata.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils_multidata.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        pos_weight_tensor = None
        if args.pos_weight is not None:
            pos_weight_tensor = torch.tensor([args.pos_weight], device=device)
            print(f"Using BCEWithLogitsLoss with pos_weight={args.pos_weight:.6f} (≈ N_neg / N_pos)")
        else:
            print("Using BCEWithLogitsLoss without pos_weight; consider setting --pos_weight=N_neg/N_pos for imbalanced splits")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils_multidata.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    if (args.dataset == 'Mumtaz' or args.dataset == 'FACED' or args.dataset == 'SEED-V' or
            args.dataset == 'PHYSIO' or args.dataset == 'SHU-MI' or args.dataset == 'ISRUC' or
            args.dataset == 'CHB-MIT' or args.dataset == 'BCIC2020-3' or args.dataset == 'SEED-VIG' or
            args.dataset == 'MentalArithmetic' or args.dataset == 'BCIC-IV-2a'):
        dataloadertype = 'CBRamode'
    else:
        dataloadertype = ''
    if args.eval:
        balanced_accuracy = []
        accuracy = []
        # for data_loader in data_loader_test:
        #     test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1), dataloadertype=dataloadertype)
        #     accuracy.append(test_stats['accuracy'])
        #     balanced_accuracy.append(test_stats['balanced_accuracy'])

        test_stats = evaluate(data_loader_val, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1), dataloadertype=dataloadertype)
        accuracy.append(test_stats['accuracy'])
        balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(f"======Accuracy val : {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        balanced_accuracy = []
        accuracy = []
        test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1), dataloadertype=dataloadertype)
        accuracy.append(test_stats['accuracy'])
        balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(f"======Accuracy test: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    max_balanced_accuracy = 0.0
    max_balanced_accuracy_test = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch<=5:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
            # for n, p in model.named_parameters():
            #     p.requires_grad = ('head' in n) or ('fc_norm' in n)
        else:
            for p in model.patch_embed.parameters():
                p.requires_grad = True
            # for p in model.parameters():
            #     p.requires_grad = True

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
            ch_names=ch_names, is_binary=args.nb_classes == 1, dataloadertype=dataloadertype
        )

        if args.output_dir and args.save_ckpt:
            utils_multidata.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
            
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1,dataloadertype=dataloadertype)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1, dataloadertype=dataloadertype)
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")
            
            if max_accuracy < val_stats["balanced_accuracy"]:
                max_accuracy = val_stats["balanced_accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils_multidata.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                max_accuracy_test = test_stats["accuracy"]
                max_balanced_accuracy = val_stats["balanced_accuracy"]
                max_balanced_accuracy_test = test_stats["balanced_accuracy"]

            print(f'Max accuracy val: {max_accuracy:.2f}%, balanced_accuracy:{max_balanced_accuracy:.2f}%, accuracy test: {max_accuracy_test:.2f}% balanced_accuracy test: {max_balanced_accuracy_test:.2f}%')
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
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils_multidata.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import torch.multiprocessing as mp

    opts, ds_init = get_args()
    spawn_required_datasets = {"PHYSIO"}
    if opts.dataset.upper() in spawn_required_datasets:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # метод уже установлен

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
