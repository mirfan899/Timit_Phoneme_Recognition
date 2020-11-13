from string import digits
import numpy as np
import python_speech_features as python_speech_features
from scipy.io import wavfile
import os
from pocketsphinx import DefaultConfig, Decoder, get_model_path, get_data_path
from constants import DICTIONARY, MAP_DICTIONARY


def generate_transcripts():
    total_lines = open("corpus/reps.txt", "r").readlines()
    total_lines = list(map(lambda x: x.strip(), total_lines))
    total_lines = list(filter(None, total_lines))
    sentences = total_lines[::3]
    uuids = total_lines[1:][::3]

    for (line, uuid) in zip(sentences, uuids):
        sentence = line.split("  ", 2)[-1]
        uuid = uuid.split(" ")[0]
        with open("kids_speech/transcripts/{}.txt".format(uuid), "w") as f:
            f.write(sentence)


def get_phonemes(sentence):
    words = sentence.upper().split()
    phonemes = []
    phonemes_only = []
    for word in words:
        if word in DICTIONARY.keys():
            phonemes.extend(DICTIONARY[word].split())
            phonemes_only.append(DICTIONARY[word])
    phonemes_only_cleaned = []
    for ph in phonemes_only:
        p = ph.translate(str.maketrans('', '', digits))
        phonemes_only_cleaned.append(p)

    phonemes = " ".join(phonemes).lower()
    phonemes = phonemes.translate(str.maketrans('', '', digits))
    result = "sil " + " ".join(phonemes) + " [sil];"

    return sentence.lower().split(), phonemes_only_cleaned, phonemes.split(), result


def get_phonemes_only(sentence):
    words = sentence.upper().split()
    phonemes = []
    phonemes_only = []
    for word in words:
        if word in DICTIONARY.keys():
            phonemes.extend(DICTIONARY[word].split())
            phonemes_only.append(DICTIONARY[word])
    phonemes_only_cleaned = []
    for ph in phonemes_only:
        p = ph.translate(str.maketrans('', '', digits))
        phonemes_only_cleaned.append(p)

    return phonemes_only_cleaned


def get_pocketsphinx_decoder(name):
    model_path = get_model_path()
    data_path = get_data_path()

    # Create a decoder with a certain model
    config = DefaultConfig()
    config.set_string('-hmm', os.path.join(model_path, 'en-us'))
    # config.set_string('-lm', os.path.join(model_path, 'en-us.lm.bin'))
    config.set_string('-jsgf', 'pocket/{}.jsgf'.format(name))
    config.set_string('-dict', 'phonemes.dict')
    config.set_float('-beam', 1e-80)
    config.set_float('-pbeam', 1e-80)
    config.set_float('-wbeam', 1e-60)
    # config.set_string('-backtrace', 'yes')
    # config.set_string('-fsgusefiller', 'no')
    # config.set_string('-bestpath', 'no')
    config.set_float('-lw', 1.0)
    decoder = Decoder(config)
    return decoder


def get_test_pocketsphinx_decoder(name):
    model_path = get_model_path()
    data_path = get_data_path()

    # Create a decoder with a certain model
    config = DefaultConfig()
    config.set_string('-hmm', os.path.join(model_path, 'en-us'))
    # config.set_string('-lm', os.path.join(model_path, 'en-us.lm.bin'))
    config.set_string('-jsgf', 'test_pocket/{}.jsgf'.format(name))
    config.set_string('-dict', 'phonemes.dict')
    # config.set_float('-beam', 1e-80)
    # config.set_float('-pbeam', 1e-80)
    # config.set_float('-wbeam', 1e-60)
    # config.set_string('-backtrace', 'yes')
    # config.set_string('-fsgusefiller', 'no')
    # config.set_string('-bestpath', 'no')
    config.set_float('-lw', 2.0)
    decoder = Decoder(config)
    return decoder


def create_mfcc(method, filename, type=2):
    """
    Perform standard preprocessing, as described by Alex Graves (2012)
    http://www.cs.toronto.edu/~graves/preprint.pdf
    Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
    [1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
    method is a dummy input!!
    """
    (rate, sample) = wavfile.read(filename)
    if sample.any():
        label = os.path.splitext(os.path.basename(filename))[0].split("-")[1]
        mfcc = python_speech_features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,
                                           preemph=0.97, appendEnergy=True)
        out = mfcc
        if type > 13:
            derivative = np.zeros(mfcc.shape)
            for i in range(1, mfcc.shape[0] - 1):
                derivative[i, :] = mfcc[i + 1, :] - mfcc[i - 1, :]

            mfcc_derivative = np.concatenate((mfcc, derivative), axis=1)
            out = mfcc_derivative
            if type > 26:
                derivative2 = np.zeros(derivative.shape)
                for i in range(1, derivative.shape[0] - 1):
                    derivative2[i, :] = derivative[i + 1, :] - derivative[i - 1, :]

                out = np.concatenate((mfcc, derivative, derivative2), axis=1)
                if type > 39:
                    derivative3 = np.zeros(derivative2.shape)
                    for i in range(1, derivative2.shape[0] - 1):
                        derivative3[i, :] = derivative2[i + 1, :] - derivative2[i - 1, :]
                    out = np.concatenate((mfcc, derivative, derivative2, derivative3), axis=1)
        return out, out.shape, label
    else:
        return [], (0, 0), "NULL"


def generate_test_mfcc(method, filename, type=2):
    """
    Perform standard preprocessing, as described by Alex Graves (2012)
    http://www.cs.toronto.edu/~graves/preprint.pdf
    Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
    [1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
    method is a dummy input!!
    """
    (rate, sample) = wavfile.read(filename)
    if sample.any():
        label = os.path.splitext(os.path.basename(filename))[0].split("-", 4)[3]
        mfcc = python_speech_features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,
                                           preemph=0.97, appendEnergy=True)
        out = mfcc
        if type > 13:
            derivative = np.zeros(mfcc.shape)
            for i in range(1, mfcc.shape[0] - 1):
                derivative[i, :] = mfcc[i + 1, :] - mfcc[i - 1, :]

            mfcc_derivative = np.concatenate((mfcc, derivative), axis=1)
            out = mfcc_derivative
            if type > 26:
                derivative2 = np.zeros(derivative.shape)
                for i in range(1, derivative.shape[0] - 1):
                    derivative2[i, :] = derivative[i + 1, :] - derivative[i - 1, :]

                out = np.concatenate((mfcc, derivative, derivative2), axis=1)
                if type > 39:
                    derivative3 = np.zeros(derivative2.shape)
                    for i in range(1, derivative2.shape[0] - 1):
                        derivative3[i, :] = derivative2[i + 1, :] - derivative2[i - 1, :]
                    out = np.concatenate((mfcc, derivative, derivative2, derivative3), axis=1)
        return out, out.shape, label
    else:
        return [], (0, 0), "NULL"


def generate_align_config(sentence, name):
    phonemes = get_phonemes_only(sentence)
    align = "sil " + " ".join([ph for ph in phonemes]).lower() + " [ sil ];"
    with open("pocket/{}.jsgf".format(name), "w") as f:
        f.writelines("""#JSGF V1.0;
grammar forcing;
public <{}> = {}""".format(name, align))


def generate_test_align_config(sentence, name):
    phonemes = get_phonemes_only(sentence)
    align = "sil " + " ".join([ph for ph in phonemes]).lower() + " [ sil ];"
    with open("test_pocket/{}.jsgf".format(name), "w") as f:
        f.writelines("""#JSGF V1.0;
grammar forcing;
public <{}> = {}""".format(name, align))


def get_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def generate_phoneme_file(path):
    name = os.path.splitext(os.path.basename(path))[0] + ".phonemes"
    lines = open(path, "r").readlines()
    phonemes = []
    for line in lines:
        start, end, ph = line.split(" ")
        phonemes.append(ph.strip())

    phonemes.remove("h#")
    phonemes.remove("h#")
    phonemes = " ".join(phonemes)
    with open("data/label/{}".format(name), "w") as f:
        f.write(phonemes)


def generate_word_file(path):
    name = os.path.splitext(os.path.basename(path))[0] + ".txt"
    lines = open(path, "r").readlines()
    words = []
    for line in lines:
        start, end, word = line.split(" ")
        words.append(word.strip())

    words = " ".join(words)
    with open("data/label/{}".format(name), "w") as f:
        f.write(words)


# def map_61_to_39():
#     MAP_DICTIONARY