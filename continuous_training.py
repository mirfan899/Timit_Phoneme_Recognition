from persephone import corpus
from persephone import corpus_reader
from persephone import rnn_ctc

kids_corpus = corpus.Corpus("fbank_and_pitch", "phonemes", "kids_speech_sample")
kids_corpus = corpus_reader.CorpusReader(kids_corpus, num_train=224, batch_size=16)
# model = rnn_ctc.Model("exp/", kids_corpus, num_layers=3, hidden_size=250)
model = rnn_ctc.Model("exp/", kids_corpus, num_layers=3, hidden_size=250)
model.train(max_epochs=50)
# model.transcribe(restore_model_path="exp/model/model_best.ckpt")
