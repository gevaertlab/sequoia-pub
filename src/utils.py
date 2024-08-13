import pandas as pd
import numpy as np
import os
import h5py
import pdb
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedGroupKFold
import torch

def custom_collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of tensors from the dataaset
    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def filter_no_features(df, feature_path, feature_name):
    print(f'Filtering WSIs that do not have {feature_name} features')
    projects = np.unique(df.tcga_project)
    all_wsis_with_features = []
    remove = []
    for proj in projects:
        wsis_with_features = os.listdir(os.path.join(feature_path, proj))
        for wsi in wsis_with_features:
            try:
                with h5py.File(os.path.join(feature_path, proj, wsi, wsi+'.h5'), "r") as f:
                    cols = list(f.keys())
                    if feature_name not in cols:
                        remove.append(wsi)
            except Exception as e:
                remove.append(wsi)        
        all_wsis_with_features += wsis_with_features
    remove += df[~df['wsi_file_name'].isin(all_wsis_with_features)].wsi_file_name.values.tolist()
    print(f'Original shape: {df.shape}')
    df = df[~df['wsi_file_name'].isin(remove)].reset_index(drop=True)
    print(f'New shape: {df.shape}')
    return df


def patient_split(dataset, random_state=0):
    """Perform patient split of any of the previously defined datasets.
    """
    patients_unique = np.unique(dataset.patient_id)
    patients_train, patients_test = train_test_split(
        patients_unique, test_size=0.2, random_state=random_state)
    patients_train, patients_val = train_test_split(
        patients_train, test_size=0.2, random_state=random_state)

    indices = np.arange(len(dataset))
    train_idx = indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] == 
                        np.array(patients_train)[np.newaxis], axis=1)]
    valid_idx = indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] == 
                        np.array(patients_val)[np.newaxis], axis=1)]
    test_idx = indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] == 
                        np.array(patients_test)[np.newaxis], axis=1)]

    return train_idx, valid_idx, test_idx


def match_patient_split(dataset, split):
    """Recover previously saved patient split
    """
    train_patients, valid_patients, test_patients = split
    indices = np.arange(len(dataset))
    train_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               train_patients[np.newaxis], axis=1)]
    valid_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               valid_patients[np.newaxis], axis=1)]
    test_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                              test_patients[np.newaxis], axis=1)]

    return train_idx, valid_idx, test_idx


def patient_kfold(dataset, n_splits=5, random_state=0, valid_size=0.1):
    """Perform cross-validation with patient split.
    """
    indices = np.arange(len(dataset))

    patients_unique = np.unique(dataset.patient_id)

    skf = KFold(n_splits, shuffle=True, random_state=random_state)
    ind = skf.split(patients_unique)

    train_idx = []
    valid_idx = []
    test_idx = []

    for k, (ind_train, ind_test) in enumerate(ind):

        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                       np.array(patients_test)[np.newaxis], axis=1)])

        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=0)
            valid_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                            np.array(patients_valid)[np.newaxis], axis=1)])

        train_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                        np.array(patients_train)[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx


def match_patient_kfold(dataset, splits):
    """Recover previously saved patient splits for cross-validation.
    """

    indices = np.arange(len(dataset))
    train_idx = []
    valid_idx = []
    test_idx = []

    for train_patients, valid_patients, test_patients in splits:

        train_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        train_patients[np.newaxis], axis=1)])
        valid_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        valid_patients[np.newaxis], axis=1)])
        test_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                       test_patients[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx

def exists(x):
    return x != None
