import editdistance
original = 'sh iy hh ae d y uh r d aa r k s uw t ih n g r iy s iy w aa sh w ao t er ao l y ih r'
predicted = 'sh iy hh ae d y er d aa r k s ow t ih d g r iy s iy w ao r s w aa t er l y ih r'
distance = editdistance.eval(original.split(), predicted.split())

print("PER")
per = ((len(original) - distance)/len(original))*100

print(100-per)
