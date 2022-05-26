# ==============================================================================
#                           APP-WIDE (GLOBAL) VARIABLES
# ==============================================================================

import os

DEBUG = True

# Path variables
API_DIR      = os.path.join(os.path.dirname(os.path.realpath("__file__")),'api')
DATA_DIR     = os.path.join(API_DIR     , 'data')
IMG_DIR      = os.path.join(DATA_DIR    , 'img')
RDB_DIR      = os.path.join(DATA_DIR    , 'database')
SVD_MDL_DIR  = os.path.join(API_DIR     , 'saved_models')
SVD_VRF_DIR  = os.path.join(SVD_MDL_DIR , 'verifiers')
SVD_DTC_DIR  = os.path.join(SVD_MDL_DIR , 'detectors')

# List with all path variables' "names"
directory_list_names = ['api root', 'data dir', 'img dir', 'rep. database',
                        'saved models', 'saved verifiers', 'saved detectors']

# Tuple with all face detector and verifier names
# detector_names = ('opencv', 'ssd', 'mtcnn', 'retinaface')
detector_names = ('retinaface') # to avoid long startups during developing

# verifier_names = ('VGG-Face', 'Facenet', 'Facenet512', 'OpenFace',
#                   'DeepFace', 'DeepID' , 'ArcFace')
verifier_names = ('ArcFace') # to avoid long startups during developing

# Other variables
models         = {}                       # stores all face verifier & detector models
supported_exts = ['.jpg', '.png', '.npy', '.webp'] # stores all supported extensions

# SQLAlchemy global variables
SQLITE_DB    = 'loki.sqlite'                                # SQLite database file name
SQLITE_DB_FP = os.path.join('api/data/database', SQLITE_DB) # full path of SQLite database

sqla_engine      = None                                     # SQLAlchemy engine object
sqla_session     = None                                     # SQLAlchemy global session object
sqla_table_names = ['person', 'representation', 'proc_files', 'proc_files_temp']

# Rodrigo specific variables (because he needs a new computer :( )
skip_model_save = True