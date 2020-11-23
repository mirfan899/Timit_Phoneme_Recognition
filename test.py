import editdistance
from utils import get_phonemes_only
original = " ".join(get_phonemes_only("HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT")).lower()

predicted = 'hh iy b iy g ae n ah k ah n f y uw s t k ah n p l ey n t ah k ah n t dh iy w ih z er w uh d v ae n ih sh p l hh ay n d dh iy k er t n aa n dh iy l eh f t'
distance = editdistance.eval(original.split(), predicted.split())

print(original)
print(predicted)
print("PER")
per = ((len(original) - distance)/len(original))*100

print(100-per)
