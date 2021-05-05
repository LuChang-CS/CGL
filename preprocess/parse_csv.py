import os
from datetime import datetime

import pandas as pd
import numpy as np


def parse_admission(path) -> dict:
    print('parsing ADMISSIONS.csv ...')
    admission_path = os.path.join(path, 'ADMISSIONS.csv')
    admissions = pd.read_csv(
        admission_path,
        usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'],
        converters={ 'SUBJECT_ID': np.int, 'HADM_ID': np.int, 'ADMITTIME': np.str }
    )
    all_patients = dict()
    for i, row in admissions.iterrows():
        if i % 100 == 0:
            print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
        pid = row['SUBJECT_ID']
        admission_id = row['HADM_ID']
        admission_time = datetime.strptime(row['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        if pid not in all_patients:
            all_patients[pid] = []
        admission = all_patients[pid]
        admission.append({
            'admission_id': admission_id,
            'admission_time': admission_time
        })
    print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

    patient_admission = dict()
    for pid, admissions in all_patients.items():
        if len(admissions) > 1:
            patient_admission[pid] = sorted(admissions, key=lambda admission: admission['admission_time'])

    return patient_admission


def parse_diagnoses(path, patient_admission: dict) -> dict:
    print('parsing DIAGNOSES_ICD.csv ...')
    diagnoses_path = os.path.join(path, 'DIAGNOSES_ICD.csv')
    diagnoses = pd.read_csv(
        diagnoses_path,
        usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'],
        converters={ 'SUBJECT_ID': np.int, 'HADM_ID': np.int, 'ICD9_CODE': np.str }
    )

    def to_standard_icd9(code: str):
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code

    admission_codes = dict()
    for i, row in diagnoses.iterrows():
        if i % 100 == 0:
            print('\r\t%d in %d rows' % (i + 1, len(diagnoses)), end='')
        pid = row['SUBJECT_ID']
        if pid in patient_admission:
            admission_id = row['HADM_ID']
            code = row['ICD9_CODE']
            if code == '':
                continue
            code = to_standard_icd9(code)
            if admission_id not in admission_codes:
                codes = []
                admission_codes[admission_id] = codes
            else:
                codes = admission_codes[admission_id]
            codes.append(code)
    print('\r\t%d in %d rows' % (len(diagnoses), len(diagnoses)))

    return admission_codes


def parse_notes(path, patient_admission: dict, use_summary=False) -> dict:
    print('parsing NOTEEVENTS.csv ...')
    notes_path = os.path.join(path, 'NOTEEVENTS.csv')
    notes = pd.read_csv(
        notes_path,
        usecols=['HADM_ID', 'TEXT', 'CATEGORY'],
        converters={'HADM_ID': lambda x: np.int(x) if x != '' else -1, 'TEXT': np.str, 'CATEGORY': np.str}
    )
    patient_note = dict()
    for i, (pid, admissions) in enumerate(patient_admission.items()):
        print('\r\t%d in %d patients' % (i + 1, len(patient_admission)), end='')
        admission_id = admissions[-1]['admission_id']
        if use_summary:
            note = [row['TEXT'] for _, row in notes[notes['HADM_ID'] == admission_id].iterrows()
                    if row['CATEGORY'] == 'Discharge summary']
        else:
            # note = notes[notes['HADM_ID'] == admission_id]['TEXT'].tolist()
            note = [row['TEXT'] for _, row in notes[notes['HADM_ID'] == admission_id].iterrows()
                    if row['CATEGORY'] != 'Discharge summary']
        note = ' '.join(note)
        if len(note) > 0:
            patient_note[pid] = note
    print('\r\t%d in %d patients' % (len(patient_admission), len(patient_admission)))
    return patient_note


def calibrate_patient_by_admission(patient_admission: dict, admission_codes: dict):
    print('calibrating patients by admission ...')
    del_pids = []
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            if admission['admission_id'] not in admission_codes:
                break
        else:
            continue
        del_pids.append(pid)
    for pid in del_pids:
        admissions = patient_admission[pid]
        for admission in admissions:
            if admission['admission_id'] in admission_codes:
                del admission_codes[admission['admission_id']]
            else:
                print('\tpatient %d have an admission %d without diagnosis' % (pid, admission['admission_id']))
        del patient_admission[pid]


def calibrate_patient_by_notes(patient_admission: dict, admission_codes: dict, patient_note: dict):
    print('calibrating patients by notes ...')
    del_pids = [pid for pid in patient_admission if pid not in patient_note]
    for pid in del_pids:
        print('\tpatient %d doesn\'t have notes' % pid)
        admissions = patient_admission[pid]
        for admission in admissions:
            del admission_codes[admission['admission_id']]
        del patient_admission[pid]
