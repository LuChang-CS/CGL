import math

import numpy as np


def split_patients(patient_admission: dict, admission_codes: dict, code_map: dict, seed=6669) -> (np.ndarray, np.ndarray, np.ndarray):
    print('splitting train, valid, and test pids')
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission['admission_id']]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    train_num = 6000
    valid_num = 125
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def build_code_xy(pids: np.ndarray,
                  patient_admission: dict,
                  admission_codes_encoded: dict,
                  max_admission_num: int,
                  code_num: int,
                  max_code_num_in_a_visit: int) -> (np.ndarray, np.ndarray, np.ndarray):
    print('building train/valid/test codes features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_admission_num, max_code_num_in_a_visit), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n, ), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['admission_id']]
            x[i][k][:len(codes)] = codes
        codes = np.array(admission_codes_encoded[admissions[-1]['admission_id']]) - 1
        y[i][codes] = 1
        lens[i] = len(admissions) - 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens


def build_time_duration_xy(pids: np.ndarray,
                           patient_time_duration_encoded: dict,
                           max_admission_num: int) -> (np.ndarray, np.ndarray):
    print('building train/valid/test time duration features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_admission_num))
    y = np.zeros((n, ))
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        duration = patient_time_duration_encoded[pid]
        x[i][:len(duration) - 1] = duration[:-1]
        y[i] = duration[-1]
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y


def build_note_x(pids: np.ndarray,
                 patient_note_encoded: dict,
                 max_word_num_in_a_note: int) -> (np.ndarray, np.ndarray):
    print('building train/valid/test notes features and labels ...')
    n = len(pids)
    x = np.zeros((n, max_word_num_in_a_note), dtype=int)
    lens = np.zeros((n, ), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        note = patient_note_encoded[pid]
        length = max_word_num_in_a_note if max_word_num_in_a_note < len(note) else len(note)
        x[i][:length] = note[:length]
        lens[i] = length
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, lens


def calculate_tf_idf(note_encoded: dict, word_num: int) -> dict:
    n_docs = len(note_encoded)
    tf = dict()
    df = np.zeros((word_num + 1, ), dtype=np.int64)
    print('calculating tf and df ...')
    for i, (pid, note) in enumerate(note_encoded.items()):
        print('\r\t%d / %d' % (i + 1, n_docs), end='')
        note_tf = dict()
        for word in note:
            note_tf[word] = note_tf.get(word, 0) + 1
        wset = set(note)
        for word in wset:
            df[word] += 1
        tf[pid] = note_tf
    print('\r\t%d / %d patients' % (n_docs, n_docs))
    print('calculating tf_idf ...')
    tf_idf = dict()
    for i, (pid, note) in enumerate(note_encoded.items()):
        print('\r\t%d / %d patients' % (i + 1, n_docs), end='')
        note_tf = tf[pid]
        note_tf_idf = [note_tf[word] / len(note) * (math.log(n_docs / (1 + df[word]), 10) + 1)
                      for word in note]
        tf_idf[pid] = note_tf_idf
    print('\r\t%d / %d patients' % (n_docs, n_docs))
    return tf_idf


def build_tf_idf_weight(pids: np.ndarray, note_x: np.ndarray, note_encoded: dict, word_num: int) -> np.ndarray:
    print('build tf_idf for notes ...')
    tf_idf = calculate_tf_idf(note_encoded, word_num)
    weight = np.zeros_like(note_x, dtype=float)
    for i, pid in enumerate(pids):
        note_tf_idf = tf_idf[pid]
        weight[i][:len(note_tf_idf)] = note_tf_idf
    weight = weight / weight.sum(axis=-1, keepdims=True)
    return weight


def build_heart_failure_y(hf_prefix: str, codes_y: np.ndarray, code_map: dict) -> np.ndarray:
    print('building train/valid/test heart failure labels ...')
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map), ), dtype=int)
    hfs[hf_list - 1] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y
