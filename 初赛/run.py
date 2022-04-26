import re
import os
import sklearn
import json
import pandas as pd
import warnings
import pefile
import multiprocessing
import lief
import ember

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from capstone import *



warnings.filterwarnings('ignore')

DEBUG = False
INITIALIZE = True

if DEBUG:
    train_sample_path = 'sample_classification_small2/train/'
    test_sample_path = 'sample_classification_small2/test/'
else:
    train_sample_path = 'sample_classification_small/train/'
    test_sample_path = 'sample_classification_small/test/'

train_sample_path_list = [train_sample_path + 'black/' + path for path in os.listdir(train_sample_path + 'black/')] + \
                         [train_sample_path + 'white/' +path for path in os.listdir(train_sample_path + 'white/')]
test_sample_path_list = [test_sample_path + path for path in os.listdir(test_sample_path)]

if DEBUG:
    train = pd.DataFrame({'file': train_sample_path_list, 'label':[1] * 10 + [0] * 20})
else:
    train = pd.DataFrame({'file': train_sample_path_list, 'label':[1] * 1000 + [0] * 2000})
test = pd.DataFrame({'file': test_sample_path_list})

print("Train files: ", len(train), "| Test files: ", len(test))

data = pd.concat([train, test]).reset_index(drop=True)
sample_list = train_sample_path_list + test_sample_path_list


def get_asm_files():
    ### 返汇编获得操作码

    os.makedirs('asm', exist_ok=True)
    for i in tqdm(range(len(sample_list))):
        try:
            pe = pefile.PE(sample_list[i])
            entrypoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            entrypoint_address = entrypoint + pe.OPTIONAL_HEADER.ImageBase
            binary_code = pe.get_memory_mapped_image()[entrypoint:]
            disassembler = Cs(CS_ARCH_X86, CS_MODE_32)

            with open('asm/' + sample_list[i].split('/')[-1] + '.txt', 'w') as f:
                for instruction in disassembler.disasm(binary_code, entrypoint_address):
                    f.writelines(instruction.mnemonic + ' ' + instruction.op_str + '\n')
        except:
            pass


def get_pe_dict():
    ### 获取PE Head

    os.makedirs('pe', exist_ok=True)
    for i in tqdm(range(len(sample_list))):
        try:
            pe = pefile.PE(sample_list[i])
            pe_dict = pe.dump_dict()
            np.save('pe/' + sample_list[i].split('/')[-1] + '.npy', pe_dict)
        except:
            pass


if INITIALIZE:
    get_asm_files()
    get_pe_dict()


def get_ByteHistogram_feature(data):
    if DEBUG:
        feature_len = 256
        feature_arr = np.zeros((len(data), feature_len))
        for i in tqdm(range(len(data))):
            with open(data['file'][i], "rb") as f:
                bytez = f.read()
            counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
            counts = np.array(counts, dtype=np.float32)
            sum_ = counts.sum()
            normalized = counts / sum_
            feature_arr[i, :] = normalized

        data[[f'ByteHistogram_{i}' for i in range(feature_len)]] = feature_arr

    elif os.path.exists('ByteHistogram.pkl'):
        data = pd.merge(data, pd.read_pickle('ByteHistogram.pkl'), on='file', how='left')
    else:
        feature_len = 256
        feature_arr = np.zeros((len(data), feature_len))
        for i in tqdm(range(len(data))):
            with open(data['file'][i], "rb") as f:
                bytez = f.read()
            counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
            counts = np.array(counts, dtype=np.float32)
            sum_ = counts.sum()
            normalized = counts / sum_
            feature_arr[i, :] = normalized

        data[[f'ByteHistogram_{i}' for i in range(feature_len)]] = feature_arr
        data[[f'ByteHistogram_{i}' for i in range(feature_len)] + ['file']].to_pickle('ByteHistogram.pkl')
    return data


def _entropy_bin_counts(block):
    # Coarse histogram, 16 bytes per bin
    # 16-bin histogram
    window = 2048
    c = np.bincount(block >> 4, minlength=16)
    p = c.astype(np.float32) / window
    # Filter non-zero elements
    wh = np.where(c)[0]
    # "* 2" b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4bits)
    H = np.sum(-p[wh] * np.log2(p[wh])) * 2
    # Up to 16 bins (max entropy is 8 bits)
    Hbin = int(H * 2)
    # Handle entropy = 8.0 bits
    if Hbin == 16:
        Hbin = 15
    return Hbin, c


def raw_features(bytez):
    window = 2048
    step = 1024
    output = np.zeros((16, 16), dtype=np.int)
    a = np.frombuffer(bytez, dtype=np.uint8)
    if a.shape[0] < window:
        Hbin, c = _entropy_bin_counts(a)
        output[Hbin, :] += c
    else:
        # Strided trick
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step, :]
        # From the blocks, compute histogram
        for block in blocks:
            Hbin, c = _entropy_bin_counts(block)
            output[Hbin, :] += c
    return output.flatten().tolist()


def get_ByteEntropyHistogram_feature(data):
    if DEBUG:
        feature_len = 256
        feature_arr = np.zeros((len(data), feature_len))
        for i in tqdm(range(len(data))):
            with open(data['file'][i], "rb") as f:
                bytez = f.read()
            window = 2048
            step = 1024
            output = np.zeros((16, 16), dtype=np.int)
            a = np.frombuffer(bytez, dtype=np.uint8)
            if a.shape[0] < window:
                Hbin, c = _entropy_bin_counts(a)
                output[Hbin, :] += c
            else:
                # Strided trick
                shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
                strides = a.strides + (a.strides[-1],)
                blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step, :]
                # From the blocks, compute histogram
                for block in blocks:
                    Hbin, c = _entropy_bin_counts(block)
                    output[Hbin, :] += c
            output = output.flatten().tolist()
            counts = np.array(output, dtype=np.float32)
            sum_ = counts.sum()
            normalized = counts / sum_
            feature_arr[i, :] = normalized

        data[[f'ByteEntropyHistogram_{i}' for i in range(feature_len)]] = feature_arr
    elif os.path.exists('ByteEntropyHistogram.pkl'):
        data = pd.merge(data, pd.read_pickle('ByteEntropyHistogram.pkl'), on='file', how='left')
    else:
        feature_len = 256
        feature_arr = np.zeros((len(data), feature_len))
        for i in tqdm(range(len(data))):
            with open(data['file'][i], "rb") as f:
                bytez = f.read()
            window = 2048
            step = 1024
            output = np.zeros((16, 16), dtype=np.int)
            a = np.frombuffer(bytez, dtype=np.uint8)
            if a.shape[0] < window:
                Hbin, c = _entropy_bin_counts(a)
                output[Hbin, :] += c
            else:
                # Strided trick
                shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
                strides = a.strides + (a.strides[-1],)
                blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step, :]
                # From the blocks, compute histogram
                for block in blocks:
                    Hbin, c = _entropy_bin_counts(block)
                    output[Hbin, :] += c
            output = output.flatten().tolist()
            counts = np.array(output, dtype=np.float32)
            sum_ = counts.sum()
            normalized = counts / sum_
            feature_arr[i, :] = normalized

        data[[f'ByteEntropyHistogram_{i}' for i in range(feature_len)]] = feature_arr
        data[[f'ByteEntropyHistogram_{i}' for i in range(feature_len)] + ['file']].to_pickle('ByteEntropyHistogram.pkl')

    return data


def raw_feature(bytez):
    _allstrings = re.compile(b'[\x20-\x7f]{5,}')
    # Occurrences of the string 'C:\', not actually extracting the path.
    _paths = re.compile(b'c:\\\\', re.IGNORECASE)
    # Occurrences of 'http://' or 'https://', not actually extracting the URLs.
    _urls = re.compile(b'https?://', re.IGNORECASE)
    # Occurrences of the string prefix 'HKEY_', not actually extracting registry names.
    _registry = re.compile(b'HKEY_')
    # Crude evidence of an MZ header (PE dropper or bubbled executable) somewhere in the byte stream
    _mz = re.compile(b'MZ')
    # all words which can read
    _words = re.compile(b"[a-zA-Z]+")
    allstrings = _allstrings.findall(bytez)
    if allstrings:
        # Statistics about strings
        string_lengths = [len(s) for s in allstrings]
        avlength = sum(string_lengths) / len(string_lengths)
        # Map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
        as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
        # Histogram count
        c = np.bincount(as_shifted_string, minlength=96)
        # Distribution of characters in printable strings (entropy)
        csum = c.sum()
        p = c.astype(np.float32) / csum
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(p[wh]))
    else:
        avlength = 0
        c = np.zeros((96,), dtype=np.float32)
        csum = 0
        H = 0
    return {
        'numstrings': len(allstrings),
        'avlength': avlength,
        'printabledist': c.tolist(),
        'printables': int(csum),
        'entropy': float(H),
        'paths': len(_paths.findall(bytez)),
        'urls': len(_urls.findall(bytez)),
        'registry': len(_registry.findall(bytez)),
        'MZ': len(_mz.findall(bytez))
    }


def get_String_feature(data):
    if DEBUG:
        feature_len = 104
        feature_arr = np.zeros((len(data), feature_len))
        for i in tqdm(range(len(data))):
            with open(data['file'][i], "rb") as f:
                bytez = f.read()
            raw = raw_feature(bytez)
            hist_divisor = float(raw['printables']) if raw['printables'] > 0 else 1.0
            feature_arr[i, :] = np.hstack([
                raw['numstrings'], raw['avlength'], raw['printables'],
                np.asarray(raw['printabledist']) / hist_divisor, raw['entropy'], raw['paths'], raw['urls'],
                raw['registry'], raw['MZ']
            ]).astype(np.float32)

        data[[f'String_{i}' for i in range(feature_len)]] = feature_arr
    elif os.path.exists('String.pkl'):
        data = pd.merge(data, pd.read_pickle('String.pkl'), on='file', how='left')
    else:
        feature_len = 104
        feature_arr = np.zeros((len(data), feature_len))
        for i in tqdm(range(len(data))):
            with open(data['file'][i], "rb") as f:
                bytez = f.read()
            raw = raw_feature(bytez)
            hist_divisor = float(raw['printables']) if raw['printables'] > 0 else 1.0
            feature_arr[i, :] = np.hstack([
                raw['numstrings'], raw['avlength'], raw['printables'],
                np.asarray(raw['printabledist']) / hist_divisor, raw['entropy'], raw['paths'], raw['urls'],
                raw['registry'], raw['MZ']
            ]).astype(np.float32)

        data[[f'String_{i}' for i in range(feature_len)]] = feature_arr
        data[[f'String_{i}' for i in range(feature_len)] + ['file']].to_pickle('String.pkl')

    return data


def get_tfidf_word(data):
    word_features = []
    for i in tqdm(range(len(data))):
        with open(data['file'][i], "rb") as f:
            bytez = f.read()
        word_features.append(get_tfidf_words(bytez=str(bytez)))
    word_feature = pd.DataFrame({'file': data['file'].values, "word_features": word_features})
    return word_feature


def get_tfidf_word2(data):
    op_code_list = []
    for i in tqdm(range(len(data))):
        asm_path = 'asm/' + data['file'][i].split('/')[-1] + '.txt'
        if os.path.exists(asm_path):
            with open(asm_path, 'r') as f:
                ss = f.readlines()
            op_code_list.append([i.split(' ')[0] for i in ss])
        else:
            op_code_list.append(['None'])
    word_feature = pd.DataFrame({'file': data['file'].values, "word_features_opcode": op_code_list})
    return word_feature


def get_tfidf_words(bytez):
    """ Extracts a list of readable strings for tf-idf """
    list_ = []
    list2 = []
    words = []

    raw_words = re.findall('[a-zA-Z]+', bytez)
    words_space = ' '.join(w for w in raw_words if 4 < len(w) < 20)
    list_.append(words_space)
    for item in list_:  # 第二轮清洗，过滤掉小于3的字符串
        if len(item) > 3:
            list2.append(item)
    for item in list2:  # 第三轮清洗,对过长的字符串进行拆分
        if len(item) > 20:
            for text in item.split():
                if (('a' in text) or ('e' in text) or ('i' in text) or ('o' in text) or ('u' in text) or (
                        'A' in text) or ('E' in text) or ('I' in text) or ('O' in text) or ('U' in text)):
                    if ('abcdef' not in text) and ('aaaaaa' not in text) and ('<init>' not in text):
                        words.append(text)

    return words


def tfidf_feature(data, index, target, min_df=10, decomposition=False, n_components=16):
    df_bag = data.copy()
    doc_list = [' '.join(i) for i in df_bag[target]]
    tfidf_vector = TfidfVectorizer(min_df=min_df).fit_transform(doc_list)
    if decomposition:

        nmf = NMF(random_state=2020, n_components=n_components)
        df_bag[[
            f'nmf_{i + 1}_{target}' for i in range(nmf.n_components)
        ]] = pd.DataFrame(nmf.fit_transform(
            tfidf_vector),
            index=df_bag.index)

        svd = TruncatedSVD(random_state=2020,
                           n_components=n_components)
        df_bag[[
            f'svd_{i + 1}_{target}' for i in range(svd.n_components)
        ]] = pd.DataFrame(svd.fit_transform(
            tfidf_vector),
            index=df_bag.index)
    else:
        df_tfidf = tfidf_vector.todense()
        print('df_tfidf:' + str(df_tfidf.shape))
        tfidf_columns = [f'tfidf_{target}_{i + 1}' for i in range(df_tfidf.shape[1])]
        df_bag[tfidf_columns] = pd.DataFrame(df_tfidf, index=df_bag.index)
    return df_bag


def get_Tfidf_feature(data):
    if os.path.exists('Tfidf.pkl'):
        data = pd.merge(data, pd.read_pickle('Tfidf.pkl'), on='file', how='left')
    else:
        word_feature = get_tfidf_word(data)
        data_tfidf = tfidf_feature(word_feature, 'file', 'word_features', decomposition=True, min_df=3, n_components=64)
        data_tfidf.drop('word_features', axis=1, inplace=True)
        data_tfidf.to_pickle('Tfidf.pkl')
        data = pd.merge(data, pd.read_pickle('Tfidf.pkl'), on='file', how='left')

    return data


def get_Tfidf_feature2(data):
    if os.path.exists('Tfidf_opcode.pkl'):
        data = pd.merge(data, pd.read_pickle('Tfidf_opcode.pkl'), on='file', how='left')
    else:
        word_feature = get_tfidf_word2(data)
        data_tfidf = tfidf_feature(word_feature, 'file', 'word_features_opcode', decomposition=True, min_df=5,
                                   n_components=32)
        data_tfidf.drop('word_features_opcode', axis=1, inplace=True)
        data_tfidf.to_pickle('Tfidf_opcode.pkl')
        data = pd.merge(data, pd.read_pickle('Tfidf_opcode.pkl'), on='file', how='left')

    return data


def get_pe_feature(data):
    if os.path.exists('pe.pkl'):
        data = pd.merge(data, pd.read_pickle('pe.pkl'), on='file', how='left')
    else:
        from collections import defaultdict
        pe_feature_dict = defaultdict(list)

        for i in tqdm(range(len(data['file'].values))):
            dict_path = 'pe/' + data['file'].values[i].split('/')[-1] + '.npy'
            if os.path.exists(dict_path):
                pe = np.load(dict_path, allow_pickle=True).item()

                ## Parsing Warnings
                if 'Parsing Warnings' in pe.keys():
                    pe_feature_dict['Parsing Warnings'].append(len(pe['Parsing Warnings']))
                else:
                    pe_feature_dict['Parsing Warnings'].append(0)

                ## DOS_HEADER
                pe_feature_dict['e_magic'].append(pe['DOS_HEADER']['e_magic']['Value'])
                pe_feature_dict['e_cblp'].append(pe['DOS_HEADER']['e_cblp']['Value'])
                pe_feature_dict['e_crlc'].append(pe['DOS_HEADER']['e_crlc']['Value'])
                pe_feature_dict['e_cparhdr'].append(pe['DOS_HEADER']['e_cparhdr']['Value'])
                pe_feature_dict['e_minalloc'].append(pe['DOS_HEADER']['e_minalloc']['Value'])
                pe_feature_dict['e_maxalloc'].append(pe['DOS_HEADER']['e_maxalloc']['Value'])
                pe_feature_dict['e_ss'].append(pe['DOS_HEADER']['e_ss']['Value'])
                pe_feature_dict['e_sp'].append(pe['DOS_HEADER']['e_sp']['Value'])
                pe_feature_dict['e_csum'].append(pe['DOS_HEADER']['e_csum']['Value'])
                pe_feature_dict['e_ip'].append(pe['DOS_HEADER']['e_ip']['Value'])
                pe_feature_dict['e_cs'].append(pe['DOS_HEADER']['e_cs']['Value'])
                pe_feature_dict['e_lfarlc'].append(pe['DOS_HEADER']['e_lfarlc']['Value'])
                pe_feature_dict['e_ovno'].append(pe['DOS_HEADER']['e_ovno']['Value'])
                pe_feature_dict['e_oemid'].append(pe['DOS_HEADER']['e_oemid']['Value'])
                pe_feature_dict['e_oeminfo'].append(pe['DOS_HEADER']['e_oeminfo']['Value'])
                pe_feature_dict['e_lfanew'].append(pe['DOS_HEADER']['e_lfanew']['Value'])

                ## NT_HEADERS
                pe_feature_dict['Signature'].append(pe['NT_HEADERS']['Signature']['Value'])

                ## FILE_HEADER
                pe_feature_dict['Machine'].append(pe['FILE_HEADER']['Machine']['Value'])
                pe_feature_dict['NumberOfSections'].append(pe['FILE_HEADER']['NumberOfSections']['Value'])
                pe_feature_dict['PointerToSymbolTable'].append(pe['FILE_HEADER']['PointerToSymbolTable']['Value'])
                pe_feature_dict['NumberOfSymbols'].append(pe['FILE_HEADER']['NumberOfSymbols']['Value'])
                pe_feature_dict['SizeOfOptionalHeader'].append(pe['FILE_HEADER']['SizeOfOptionalHeader']['Value'])
                pe_feature_dict['Characteristics'].append(pe['FILE_HEADER']['Characteristics']['Value'])

                ## Flags
                pe_feature_dict['Flags'].append(len(pe['Flags']))

                ## OPTIONAL_HEADER
                pe_feature_dict['Magic'].append(pe['OPTIONAL_HEADER']['Magic']['Value'])
                pe_feature_dict['MajorLinkerVersion'].append(pe['OPTIONAL_HEADER']['MajorLinkerVersion']['Value'])
                pe_feature_dict['MinorLinkerVersion'].append(pe['OPTIONAL_HEADER']['MinorLinkerVersion']['Value'])
                pe_feature_dict['SizeOfCode'].append(pe['OPTIONAL_HEADER']['SizeOfCode']['Value'])
                pe_feature_dict['SizeOfInitializedData'].append(pe['OPTIONAL_HEADER']['SizeOfInitializedData']['Value'])
                pe_feature_dict['SizeOfUninitializedData'].append(
                    pe['OPTIONAL_HEADER']['SizeOfUninitializedData']['Value'])
                pe_feature_dict['AddressOfEntryPoint'].append(pe['OPTIONAL_HEADER']['AddressOfEntryPoint']['Value'])
                pe_feature_dict['BaseOfCode'].append(pe['OPTIONAL_HEADER']['BaseOfCode']['Value'])

                if 'BaseOfData' in pe['OPTIONAL_HEADER'].keys():
                    pe_feature_dict['BaseOfData'].append(pe['OPTIONAL_HEADER']['BaseOfData']['Value'])
                else:
                    pe_feature_dict['BaseOfData'].append(0)

                pe_feature_dict['ImageBase'].append(pe['OPTIONAL_HEADER']['ImageBase']['Value'])
                pe_feature_dict['SectionAlignment'].append(pe['OPTIONAL_HEADER']['SectionAlignment']['Value'])
                pe_feature_dict['FileAlignment'].append(pe['OPTIONAL_HEADER']['FileAlignment']['Value'])
                pe_feature_dict['MajorOperatingSystemVersion'].append(
                    pe['OPTIONAL_HEADER']['MajorOperatingSystemVersion']['Value'])
                pe_feature_dict['MinorOperatingSystemVersion'].append(
                    pe['OPTIONAL_HEADER']['MinorOperatingSystemVersion']['Value'])
                pe_feature_dict['MajorImageVersion'].append(pe['OPTIONAL_HEADER']['MajorImageVersion']['Value'])
                pe_feature_dict['MinorImageVersion'].append(pe['OPTIONAL_HEADER']['MinorImageVersion']['Value'])
                pe_feature_dict['Reserved1'].append(pe['OPTIONAL_HEADER']['Reserved1']['Value'])
                pe_feature_dict['SizeOfImage'].append(pe['OPTIONAL_HEADER']['SizeOfImage']['Value'])
                pe_feature_dict['SizeOfHeaders'].append(pe['OPTIONAL_HEADER']['SizeOfHeaders']['Value'])
                pe_feature_dict['CheckSum'].append(pe['OPTIONAL_HEADER']['CheckSum']['Value'])
                pe_feature_dict['Subsystem'].append(pe['OPTIONAL_HEADER']['Subsystem']['Value'])
                pe_feature_dict['DllCharacteristics'].append(pe['OPTIONAL_HEADER']['DllCharacteristics']['Value'])
                pe_feature_dict['SizeOfStackReserve'].append(pe['OPTIONAL_HEADER']['SizeOfStackReserve']['Value'])
                pe_feature_dict['SizeOfStackCommit'].append(pe['OPTIONAL_HEADER']['SizeOfStackCommit']['Value'])
                pe_feature_dict['SizeOfHeapReserve'].append(pe['OPTIONAL_HEADER']['SizeOfHeapReserve']['Value'])
                pe_feature_dict['SizeOfHeapCommit'].append(pe['OPTIONAL_HEADER']['SizeOfHeapCommit']['Value'])
                pe_feature_dict['LoaderFlags'].append(pe['OPTIONAL_HEADER']['LoaderFlags']['Value'])
                pe_feature_dict['NumberOfRvaAndSizes'].append(pe['OPTIONAL_HEADER']['NumberOfRvaAndSizes']['Value'])

                ## DllCharacteristics
                pe_feature_dict['DllCharacteristics_len'].append(len(pe['DllCharacteristics']))

                ## PE Sections
                Entropy_list = []
                for sec in range(pe['FILE_HEADER']['NumberOfSections']['Value']):
                    Entropy_list.append(pe['PE Sections'][sec]['Entropy'])
                pe_feature_dict['Entropy_mean'].append(np.mean(Entropy_list))
                pe_feature_dict['Entropy_std'].append(np.std(Entropy_list))
                pe_feature_dict['Entropy_max'].append(np.max(Entropy_list))
                pe_feature_dict['Entropy_min'].append(np.min(Entropy_list))

                ## Directories
                pe_feature_dict['Directories_len'].append(len(pe['Directories']))

                ## Imported symbols
                if 'Imported symbols' in pe.keys():

                    Name_list = re.findall("'Name': b'([\S]*)',", str(pe['Imported symbols']))
                    pe_feature_dict['Imported symbols list'].append(Name_list)

                else:
                    pe_feature_dict['Imported symbols list'].append(['None'])
            else:
                for k in pe_feature_dict.keys():
                    if k == 'Imported symbols list':
                        pe_feature_dict[k].append(['None'])
                    else:
                        pe_feature_dict[k].append(-1)

        pe_feature = pd.DataFrame(pe_feature_dict)
        pe_feature['file'] = list(data['file'])
        pe_feature.to_pickle('pe.pkl')
        data = pd.merge(data, pd.read_pickle('pe.pkl'), on='file', how='left')

    return data


def get_w2v_word(data):
    op_code_list = []
    for i in tqdm(range(len(data))):
        asm_path = 'asm/' + data['file'][i].split('/')[-1] + '.txt'
        if os.path.exists(asm_path):
            with open(asm_path, 'r') as f:
                ss = f.readlines()
            op_code_list.append([i.replace('\n', ' ').split(' ') for i in ss][:5000])
        else:
            op_code_list.append(['None'])
    word_feature = pd.DataFrame({'file': data['file'].values, "word_features_opcode": op_code_list})
    print(word_feature.head())
    return word_feature


def w2v_feature(data, target, emb_size=64):
    tmp = data.copy()
    sentences = tmp[target].values

    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=10, min_count=3, sg=0, hs=0, seed=1, iter=5)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv.vocab:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_emb_{}'.format(target, i)] = emb_matrix[:, i]
    return tmp


def get_w2v_feature(data):
    if os.path.exists('W2v_opcode.pkl'):
        data = pd.merge(data, pd.read_pickle('W2v_opcode.pkl'), on='file', how='left')
    else:
        word_feature = get_w2v_word(data)
        data_w2v = w2v_feature(word_feature, 'word_features_opcode')
        data_w2v.drop('word_features_opcode', axis=1, inplace=True)
        data_w2v.to_pickle('W2v_opcode.pkl')
        data = pd.merge(data, pd.read_pickle('W2v_opcode.pkl'), on='file', how='left')
    return data


def get_Tfidf_feature3(data):
    if os.path.exists('Tfidf_import.pkl'):
        data = pd.merge(data, pd.read_pickle('Tfidf_import.pkl'), on='file', how='left')
    else:
        word_feature = data[['file', 'Imported symbols list']]
        data_tfidf = tfidf_feature(word_feature, 'file', 'Imported symbols list', decomposition=True, min_df=1,
                                   n_components=32)
        data_tfidf.drop('Imported symbols list', axis=1, inplace=True)
        data_tfidf.to_pickle('Tfidf_import.pkl')
        data = pd.merge(data, pd.read_pickle('Tfidf_import.pkl'), on='file', how='left')

    return data


def get_ember_feature(data):
    if os.path.exists('ember.pkl'):
        data = pd.merge(data, pd.read_pickle('ember.pkl'), on='file', how='left')
    else:
        emb_feature_list = []
        extractor = ember.PEFeatureExtractor(2)
        for sample in tqdm(data['file'].values):
            f = extractor.feature_vector(open(sample, "rb").read())
            emb_feature_list.append(f)
        emb_f = pd.DataFrame(emb_feature_list)
        emb_f.columns = [f'ember_f_{f}' for f in range(2381)]
        emb_f['file'] = list(data['file'])
        emb_f.to_pickle('ember.pkl')
        data = pd.merge(data, pd.read_pickle('ember.pkl'), on='file', how='left')
    return data

##### 特征提取

# 字节直方图
data = get_ByteHistogram_feature(data)
# 字节熵直方图
data = get_ByteEntropyHistogram_feature(data)
# 字符串信息
data = get_String_feature(data)
# 可读性字符串
data = get_Tfidf_feature(data)
# PE 头文件信息
data = get_pe_feature(data)
# 操作码
data = get_Tfidf_feature2(data)
# 操作码
data = get_w2v_feature(data)
# 导包信息
data = get_Tfidf_feature3(data)
# ember 特征
# Anderson HS, Roth P. Ember: an open dataset for training static pe malware machinelearning models[J]. arXiv preprint arXiv:1804.04637, 2018.
data = get_ember_feature(data)

##### 分离训练集测试集
train = data[~data['label'].isna()].reset_index(drop=True)
test = data[data['label'].isna()].reset_index(drop=True)

features = [i for i in train.columns if i not in ['file', 'label', 'Imported symbols list']]
y = train['label']
print("Train files: ", len(train), "| Test files: ", len(test), "| Feature nums", len(features))


def train_model(X_train, X_test, features, y, save_model=False):
    """
    训练lgb模型
    """
    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 2 ** 6,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 5000,
        'max_bin': 255,
        'verbose': -1,
        'seed': 2021,
        'bagging_seed': 2021,
        'feature_fraction_seed': 2021,
        'early_stopping_rounds': 100,
    }
    oof_lgb = np.zeros(len(X_train))
    predictions_lgb = np.zeros((len(X_test)))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
        trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
        num_round = 10000
        clf = lgb.train(
            params,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=100,
            early_stopping_rounds=50,
        )

        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration) / 5
        feat_imp_df['imp'] += clf.feature_importance() / 5
        if save_model:
            clf.save_model(f'model_{fold_}.txt')

    print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Fbeta score: {}".format(fbeta_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb], beta=0.5)))


    return feat_imp_df, oof_lgb, predictions_lgb

feat_imp_df, oof_lgb, predictions_lgb = train_model(train, test, features, y)

###### 阈值后处理
best_score = 0
for thre in tqdm(range(1, 100)):
    score = fbeta_score(y, [1 if i >= thre / 100 else 0 for i in oof_lgb], beta=0.5)
    if score > best_score:
        best_thre = thre / 100
        best_score = score

print("best_score: ", best_score, "best_thre: ", best_thre)

###### 生成提交文件
test['file_name'] = test['file'].apply(lambda x: x.split('/')[-1])
test['label'] = [1 if i >= best_thre else 0 for i in predictions_lgb]
test[['file_name', 'label']].to_csv('sub.csv', header=None, index=False, sep='|')
test['label'].sum()
