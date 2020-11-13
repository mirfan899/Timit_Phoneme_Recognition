import glob
import os
from constants import MAP_DICTIONARY
from utils import get_phonemes_only, get_name_from_path, generate_phoneme_file, generate_word_file, \
    generate_cmu_phoneme_file


def generate_phoneme_files():
    transcripts = glob.glob("kids_speech_sample/label/*.txt")
    for transcript in transcripts:
        name = get_name_from_path(transcript)
        text = open(transcript).readlines()[0]
        phonemes = get_phonemes_only(text)
        with open("kids_speech_sample/transcripts/{}".format(name + ".phonemes"), "w") as f:
            f.write(" ".join(phonemes))


def generate_timit_data():
    root = 'TIMIT'
    phonemes = {}
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if "PHN" in file:
                generate_phoneme_file(os.path.join(subdir, file))
                phonemes.update()
            elif "WRD" in file:
                generate_word_file(os.path.join(subdir, file))
            if "WAV" in file:
                file.replace("WAV", 'wav')
                os.system("cp {} ./timit/wav/".format(os.path.join(subdir, file)))
            # print(os.path.join(subdir, file))


def get_phoneme_list():
    files = glob.glob('timit/label/*.phonemes')
    phonemes = []

    for file in files:
        lines = open(file, "r").readlines()
        for line in lines:
            phones = line.split(" ")
            phonemes.extend(phones)
    return set(phonemes)


def generate_cmu_phonemes():
    files = glob.glob('timit/label/*.txt')
    for file in files:
        generate_cmu_phoneme_file(file)


if __name__ == "__main__":
    # generate_phoneme_files()
    # generate_timit_data()
    phonemes = get_phoneme_list()
    print(list(phonemes))
    print(len(phonemes))
    print([MAP_DICTIONARY[el] for el in phonemes])
