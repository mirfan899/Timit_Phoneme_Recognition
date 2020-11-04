from persephone import corpus
from persephone import experiment

corp = corpus.Corpus("fbank_and_pitch", "phonemes", "kids_speech_sample")
experiment.train_ready(corp)
