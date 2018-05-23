from notes import NoteSeqs
import os, pickle, sys
import numpy as np


name = sys.argv[1]

dataset = []
root = f'dataset/{name}'

for fname in os.listdir(root):
    path = os.path.join(root, fname)
    if os.path.isfile(path) and path.lower().endswith('.mid'):
        try:
            ns = NoteSeqs(path)
            print(path)
            array = ns.get_piano_roll().astype(np.int8)
            dataset.append(array)
        except KeyboardInterrupt:
            break
        except:
            pass

pickle.dump(dataset, open(f'dataset/{name}.pkl', 'wb'))
