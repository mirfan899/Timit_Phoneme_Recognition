from persephone import corpus
from persephone import corpus_reader
from persephone import rnn_ctc

lt_corpus = corpus.Corpus("fbank", "phonemes", "lt")
lt_corpus = corpus_reader.CorpusReader(lt_corpus, batch_size=32)
model = rnn_ctc.Model("exp_fbank/", lt_corpus, num_layers=3, hidden_size=256)
# model.train(max_epochs=50)
# model.eval(restore_model_path="exp_fbank/model/model_best.ckpt")
model.transcribe(restore_model_path="exp_fbank/model/model_best.ckpt")
