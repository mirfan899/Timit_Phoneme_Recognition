### Phoneme recognition on Kids speech dataset
Dataset is taken from
[kids speech](https://www.isip.piconepress.com/projects/speech/databases/kids_speech/)

Timit data added for testing.

Move .WAV files to .wav. Persephone can't handle extensions with capital letters.
```shell script
for file in *.WAV; do
    filename=$(basename -- "$file")
    name="${filename%.*}"
    mv "$file" "$(basename "$name").wav"
done
```

For more information on timit experiments and some preprocessing for reduced phonemes and PER
[Phoneme Info on Timit](https://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database)

Latest algorithm
https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md