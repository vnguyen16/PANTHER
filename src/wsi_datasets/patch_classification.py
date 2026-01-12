from __future__ import print_function, division
import os
from os.path import join as j_
import torch
import numpy as np
import pandas as pd
import h5py
import sys
from torch.utils.data import Dataset

sys.path.append('../')
from utils.pandas_helper_funcs import df_sdir, series_diff


class PatchClassificationDataset(Dataset):
    """Patch-level classification dataset based on .h5 features per slide."""

    def __init__(self,
                 df,
                 data_source,
                 target_transform=None,
                 slide_col='slide_id',
                 target_col='label',
                 label_map=None,
                 **kwargs):
        self.data_source = []
        for src in data_source:
            assert os.path.basename(src) == 'feats_h5'
            self.use_h5 = True
            self.data_source.append(src)

        self.data_df = df.copy()
        self.slide_col = slide_col
        self.target_col = target_col
        self.label_map = label_map
        self.target_transform = target_transform

        self.data_df[slide_col] = self.data_df[slide_col].astype(str)
        self.X = None
        self.y = None

        self.validate_classification_dataset()
        self.set_feat_paths_in_df()

        self.data_df.index = self.data_df[slide_col].astype(str)
        self.data_df.index.name = 'slide_id'

        # Create index mapping: global patch index → (slide_id, local_patch_idx)
        self.sample_to_patch_map = []
        self.patch_labels = []

        for slide_id in self.data_df.index:
            fpath = self.data_df.loc[slide_id, 'fpath']
            label = self.data_df.loc[slide_id, self.target_col]

            if self.label_map is not None:
                label = self.label_map[label]
            if self.target_transform is not None:
                label = self.target_transform(label)

            with h5py.File(fpath, 'r') as f:
                n_patches = f['features'].shape[0]

            for patch_idx in range(n_patches):
                self.sample_to_patch_map.append((slide_id, patch_idx))
                self.patch_labels.append(label)

        self.patch_labels = torch.LongTensor(self.patch_labels)
        print(f"Total patches: {len(self.sample_to_patch_map)}")

    def __len__(self):
        return len(self.sample_to_patch_map)

    def validate_classification_dataset(self):
        num_unique_target_labels = self.data_df.groupby(self.slide_col)[self.target_col].nunique()
        try:
            assert (num_unique_target_labels == 1).all()
        except AssertionError:
            print('Each slide must have only one unique label.')
            raise

    def set_feat_paths_in_df(self):
        self.feats_df = pd.concat([
            df_sdir(feats_dir, cols=['fpath', 'fname', self.slide_col])
            for feats_dir in self.data_source
        ]).drop(['fname'], axis=1).reset_index(drop=True)

        missing_feats = series_diff(self.data_df[self.slide_col], self.feats_df[self.slide_col])
        if len(missing_feats) > 0:
            print(f"Missing Features in Split:\n{missing_feats}")
            sys.exit()

        self.data_df = self.data_df.merge(self.feats_df, how='left', on=self.slide_col, validate='1:1')

        try:
            assert self.feats_df[self.slide_col].duplicated().sum() == 0
        except:
            print("❌ Duplicated features detected!")
            print(self.feats_df[self.feats_df[self.slide_col].duplicated()])
            sys.exit()

    def __getitem__(self, idx):
        slide_id, patch_idx = self.sample_to_patch_map[idx]
        fpath = self.data_df.loc[slide_id, 'fpath']

        with h5py.File(fpath, 'r') as f:
            feature = torch.from_numpy(f['features'][patch_idx])
            coords = f['coords'][patch_idx]

        label = self.patch_labels[idx]

        return {
            'img': feature,  # patch-level feature vector
            'coords': coords,  # patch location
            'label': label,
            'slide_id': slide_id,
            'patch_idx': patch_idx
        }
