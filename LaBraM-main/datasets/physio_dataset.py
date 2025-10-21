#import torch
from torch.utils.data import Dataset, DataLoader
#import numpy as np
#from cbr_utils.util import to_tensor
#import os
#import random
#import lmdb
#import pickle

import lmdb, pickle, os, numpy as np
from cbr_utils.util import to_tensor
#from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.db = None
        self.keys = None
        # не открываем LMDB здесь!

    def _ensure_open(self):
        if self.db is None:
            # безопасные для чтения параметры:
            self.db = lmdb.open(
                self.data_dir,
                readonly=True,
                lock=False,
                readahead=False,   # было True — часто хуже при fork
                meminit=False,
                max_readers=2048
            )
            with self.db.begin(write=False) as txn:
                all_keys = pickle.loads(txn.get(b'__keys__'))
                self.keys = all_keys[self.mode]

    def __len__(self):
        self._ensure_open()
        return len(self.keys)

    def __getitem__(self, idx):
        self._ensure_open()
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']

        # Вместо деления здесь — вернём сырые данные.
        # Модель уже делит на 100 в engine (см. ниже).
        return data, key, label

    def collate(self, batch):
        # x_data = np.array([x[0] for x in batch])
        # y_label = np.array([x[1] for x in batch])
        # return to_tensor(x_data), to_tensor(y_label).long()
        xs, files, ys = zip(*batch)
        return to_tensor(xs), list(files), to_tensor(ys).long()

    # Чтобы объект датасета можно было безоп. сериализовать воркерам:
    def __getstate__(self):
        state = self.__dict__.copy()
        state['db'] = None  # не передаём открытое окружение через pickle
        return state


#
# class CustomDataset(Dataset):
#     def __init__(
#             self,
#             data_dir,
#             mode='train',
#     ):
#         super(CustomDataset, self).__init__()
#         self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
#         with self.db.begin(write=False) as txn:
#             self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
#
#     def __len__(self):
#         return len((self.keys))
#
#     def __getitem__(self, idx):
#         key = self.keys[idx]
#         with self.db.begin(write=False) as txn:
#             pair = pickle.loads(txn.get(key.encode()))
#         data = pair['sample']
#         label = pair['label']
#         # предполагаю что проблема в работе лабрам изза входных данных отличающихся от 5 секунд. добавляем 5ю секунду как копию 4й
#         if data.shape!=(64,4,200):
#             print("data.shape:", data.shape)
#       #  data=data[:16, :, :]
#     #    last_second = data[:, -1:, :]
#      #   data = np.concatenate([data, last_second], axis=1)
#
#         # print(key)
#         # print(data)
#         # print(label)
#         return data/100, label
#
#     def collate(self, batch):
#         x_data = np.array([x[0] for x in batch])
#         y_label = np.array([x[1] for x in batch])
#         return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        if 'labram' in self.params.model:   # if Labram - it should be labram_base_patch200_200
            return train_set, val_set, test_set
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
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
