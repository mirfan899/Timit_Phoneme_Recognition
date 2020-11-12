from persephone import corpus
from persephone import corpus_reader
from persephone import rnn_ctc

timit_corpus = corpus.Corpus("fbank_and_pitch", "phonemes", "timit")
timit_corpus = corpus_reader.CorpusReader(timit_corpus, batch_size=32)
# model = rnn_ctc.Model("exp/", kids_corpus, num_layers=3, hidden_size=250)
model = rnn_ctc.Model("exp/", timit_corpus, num_layers=3, hidden_size=256)
# model.train(max_epochs=50)
model.eval(restore_model_path="exp/model/model_best.ckpt")
# model.transcribe(restore_model_path="exp/model/model_best.ckpt")
