import json
librispeech = open("librispeech-lexicon.json", "r")
DICTIONARY = json.load(librispeech)
