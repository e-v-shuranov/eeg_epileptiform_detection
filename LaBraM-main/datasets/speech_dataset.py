import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from cbr_utils.util import to_tensor
import os
import random
import lmdb
import pickle

class SpeechAugmentor:
    """Applies lightweight stochastic augmentations for imagined-speech EEG."""

    def __init__(
        self,
        jitter_std: float = 0.0,
        jitter_prob: float = 0.0,
        time_shift_prob: float = 0.0,
        time_shift_max_pct: float = 0.0,
        channel_dropout_prob: float = 0.0,
        channel_dropout_max_pct: float = 0.0,
    ) -> None:
        self.jitter_std = max(0.0, float(jitter_std))
        self.jitter_prob = max(0.0, float(jitter_prob))
        self.time_shift_prob = max(0.0, float(time_shift_prob))
        self.time_shift_max_pct = max(0.0, float(time_shift_max_pct))
        self.channel_dropout_prob = max(0.0, float(channel_dropout_prob))
        self.channel_dropout_max_pct = max(0.0, float(channel_dropout_max_pct))

    def _maybe_apply_jitter(self, data: np.ndarray) -> np.ndarray:
        if self.jitter_prob <= 0.0 or self.jitter_std <= 0.0:
            return data
        if random.random() >= self.jitter_prob:
            return data
        noise_scale = np.std(data, axis=-1, keepdims=True)
        noise_scale = np.maximum(noise_scale, 1e-6)
        noise_scale = noise_scale.astype(data.dtype, copy=False)
        noise = np.random.normal(loc=0.0, scale=self.jitter_std, size=data.shape).astype(data.dtype, copy=False)
        return data + noise * noise_scale

    def _maybe_apply_time_shift(self, data: np.ndarray) -> np.ndarray:
        if self.time_shift_prob <= 0.0 or self.time_shift_max_pct <= 0.0:
            return data
        if random.random() >= self.time_shift_prob:
            return data
        time_len = data.shape[-1]
        max_shift = int(self.time_shift_max_pct * time_len)
        if max_shift <= 0:
            return data
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return data
        return np.roll(data, shift=shift, axis=-1)

    def _maybe_apply_channel_dropout(self, data: np.ndarray) -> np.ndarray:
        if self.channel_dropout_prob <= 0.0 or self.channel_dropout_max_pct <= 0.0:
            return data
        if data.ndim < 2 or random.random() >= self.channel_dropout_prob:
            return data
        num_channels = data.shape[0]
        drop_count = max(1, int(round(self.channel_dropout_max_pct * num_channels)))
        drop_count = min(drop_count, num_channels)
        if drop_count <= 0:
            return data
        drop_idx = np.random.choice(num_channels, size=drop_count, replace=False)
        data = data.copy()
        data[drop_idx, ...] = 0.0
        return data

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        augmented = np.array(sample, copy=True)
        augmented = self._maybe_apply_time_shift(augmented)
        augmented = self._maybe_apply_channel_dropout(augmented)
        augmented = self._maybe_apply_jitter(augmented)
        return augmented


def _build_speech_augmentor(params):
    jitter_std = getattr(params, 'speech_jitter_std', 0.0)
    jitter_prob = getattr(params, 'speech_jitter_prob', 0.0)
    time_shift_prob = getattr(params, 'speech_time_shift_prob', 0.0)
    time_shift_pct = getattr(params, 'speech_time_shift_pct', 0.0)
    channel_drop_prob = getattr(params, 'speech_channel_dropout_prob', 0.0)
    channel_drop_pct = getattr(params, 'speech_channel_dropout_max_pct', 0.0)

    if max(jitter_std, jitter_prob, time_shift_prob, time_shift_pct, channel_drop_prob, channel_drop_pct) <= 0:
        return None

    return SpeechAugmentor(
        jitter_std=jitter_std,
        jitter_prob=jitter_prob,
        time_shift_prob=time_shift_prob,
        time_shift_max_pct=time_shift_pct,
        channel_dropout_prob=channel_drop_prob,
        channel_dropout_max_pct=channel_drop_pct,
    )


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            augmentor=None,
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        self.augmentor = augmentor if mode == 'train' else None

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        if self.augmentor is not None:
            data = self.augmentor(data)
        label =  int(pair['label'])
        # print(key)
        # print(data.shape)
        # print(label)
        return data, key, label

    def collate(self, batch):
        xs, files, ys = zip(*batch)
        return to_tensor(np.array(xs)), list(files), to_tensor(np.array(ys)).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        augmentor = _build_speech_augmentor(self.params)
        train_set = CustomDataset(self.datasets_dir, mode='train', augmentor=augmentor)
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        if 'labram' in getattr(self.params, 'model', '').lower():
            return train_set, val_set, test_set

        print(len(train_set), len(val_set), len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
