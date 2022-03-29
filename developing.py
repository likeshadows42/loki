import os
import numpy                as np
import IFR.classes          as cls
import IFR.functions        as funcs
import api.global_variables as glb

from tqdm                   import tqdm
from deepface.DeepFace      import build_model         as build_verifier

verifier_names = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace',
                  'DeepFace', 'DeepID' , 'ArcFace'   , 'Dlib']

zip_fp = '/home/rpessoa/projects/loki/api/test_zip.7z'


# pbar = tqdm(range(0, len(glb.verifier_names)), desc='Loading verifiers',
#             disable = False)


# Tries to load a database if it exists. If not, create a new one.
print('  -> Loading / creating database:')
if not os.path.isfile(os.path.join(glb.RDB_DIR, 'rep_database.pickle')):
    glb.db_changed = True
glb.rep_db = funcs.load_representation_db(os.path.join(glb.RDB_DIR,
                                          'rep_database.pickle'), verbose=True)
print('')

# 
all_tags = []
for rep in glb.rep_db:
    all_tags.append(rep.name_tag)

all_tags = np.unique(all_tags)
all_tags.sort()

print('\n', all_tags, sep='')




