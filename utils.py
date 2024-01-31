import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedGroupKFold

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


def grouped_strat_split(df, n_splits1=5, n_splits2=10, random_state=0, strat_col='tcga_project'):
    """
    Grouped stratified split (grouped by patient id, strat by tcga_project)
    n_splits1 is for train/test split
    n_splits2 is for train/val split (only one fold is used in this case)
    """
    
    cv = StratifiedGroupKFold(n_splits1, shuffle=True, random_state=random_state)

    train_idx = []
    valid_idx = []
    test_idx = []

    for ind_train, ind_test in cv.split(X=df.index, y=df[strat_col], groups=df.patient_id):
        train_df_complete = df.iloc[ind_train]
        test_idx.append(ind_test)

        cv2 = StratifiedGroupKFold(n_splits2, shuffle=True, random_state=random_state) 

        for k, (ind_train, ind_val) in enumerate(cv2.split(X=train_df_complete.index, y=train_df_complete.tcga_project, groups=train_df_complete.patient_id)):
            if k == 0:
                train_idx.append(train_df_complete.index[ind_train])
                valid_idx.append(train_df_complete.index[ind_val])

    return train_idx, valid_idx, test_idx

    