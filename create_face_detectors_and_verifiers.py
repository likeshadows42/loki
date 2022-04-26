# ==============================================================================
#                FACE DETECTOR & VERIFIER MODELS CREATION SCRIPT
# ==============================================================================

# Importing necessary functions and modules
import IFR.api              as apif
import IFR.functions        as funcs

from tqdm                   import tqdm

# All face detector and verifier names
detector_names = ['opencv', 'ssd', 'mtcnn', 'retinaface']
verifier_names = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace',
                  'DeepFace', 'DeepID' , 'ArcFace']

# Save directories for the face detector and verifiers
save_dir1 = '/home/rpessoa/projects/loki/api/saved_models/detectors'
save_dir2 = '/home/rpessoa/projects/loki/api/saved_models/verifiers'

# Builds all face detectors and verifiers
detectors = funcs.batch_build_detectors(detector_names, show_prog_bar=True,
                                        verbose=False)
verifiers = funcs.batch_build_verifiers(verifier_names, show_prog_bar=True,
                                        verbose=False)

# Prints the number of face detectors and verifiers built
print('Number of detectors built:', len(detectors))
print('Number of verifiers built:', len(verifiers), '\n')

# Determines the number of face detectors and verifiers
n_detectors = len(detector_names)
n_verifiers = len(verifier_names)

# Saves each face detector model
for name, obj in zip(detector_names, detectors):
    status = apif.save_built_model(name, obj, save_dir1, overwrite=False,
                                    verbose=True)
print('')

# Saves each face verifier model
for name, obj in zip(verifier_names, verifiers):
    status = apif.save_built_model(name, obj, save_dir2, overwrite=False,
                                    verbose=True)
print('')