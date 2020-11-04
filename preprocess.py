import glob
from utils import get_phonemes_only, get_name_from_path


def generate_phoneme_files():
    transcripts = glob.glob("kids_speech_sample/label/*.txt")
    for transcript in transcripts:
        name = get_name_from_path(transcript)
        text = open(transcript).readlines()[0]
        phonemes = get_phonemes_only(text)
        with open("kids_speech_sample/transcripts/{}".format(name + ".phonemes"), "w") as f:
            f.write(" ".join(phonemes))


if __name__ == "__main__":
    generate_phoneme_files()
