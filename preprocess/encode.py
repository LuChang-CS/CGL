import re

from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))


def encode_code(admission_codes: dict) -> (dict, dict):
    print('encoding code ...')
    code_map = dict()
    for i, (admission_id, codes) in enumerate(admission_codes.items()):
        for code in codes:
            if code not in code_map:
                code_map[code] = len(code_map) + 1

    admission_codes_encoded = {
        admission_id: [code_map[code] for code in codes]
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map


def encode_time_duration(patient_admission: dict) -> dict:
    print('encoding time duration ...')
    patient_time_duration_encoded = dict()
    for pid, admissions in patient_admission.items():
        duration = [0]
        for i in range(1, len(admissions)):
            days = (admissions[i]['admission_time'] - admissions[i - 1]['admission_time']).days
            duration.append(days)
        patient_time_duration_encoded[pid] = duration
    return patient_time_duration_encoded


def extract_word(text: str) -> list:
    """Extract words from a text
    @param: text, str
    @param: max_len, the maximum length of text we want to extract, default None
    @return: list, words list in the text
    """
    # replace non-word-character with space
    text = re.sub(r'[^A-Za-z_]', ' ', text.strip().lower())
    # tokenize text using NLTK
    words = word_tokenize(text)
    clean_words = []
    for word in words:
        if word not in stopwords_set:
            word = ps.stem(word).lower()
            if word not in stopwords_set:
                clean_words.append(word)
    return clean_words


def encode_note_train(patient_note: dict, pids: np.ndarray, max_note_len=None) -> (dict, dict):
    print('encoding train notes ...')
    dictionary = dict()
    patient_note_encoded = dict()
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        words = extract_word(patient_note[pid])
        note_encoded = []
        for word in words:
            if word not in dictionary:
                wid = len(dictionary) + 1
                dictionary[word] = wid
            else:
                wid = dictionary[word]
            note_encoded.append(wid)
        if max_note_len is not None:
            note_encoded = note_encoded[:max_note_len]
        patient_note_encoded[pid] = note_encoded
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return patient_note_encoded, dictionary


def encode_note_test(patient_note: dict, pids: np.ndarray, dictionary: dict, max_note_len=None) -> dict:
    print('encoding valid/test notes ...')
    patient_note_encoded = dict()
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        words = extract_word(patient_note[pid])
        note_encoded = []
        for word in words:
            if word in dictionary:
                note_encoded.append(dictionary[word])
        if len(note_encoded) == 0:
            note_encoded.append(0)
        if max_note_len is not None:
            note_encoded = note_encoded[:max_note_len]
        patient_note_encoded[pid] = note_encoded
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return patient_note_encoded
