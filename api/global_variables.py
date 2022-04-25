# ==============================================================================
#                           APP-WIDE (GLOBAL) VARIABLES
# ==============================================================================

import os

from IFR.classes import RepDatabase

DEBUG = True

# Path variables
API_DIR      = os.path.join(os.path.dirname(os.path.realpath("__file__")),'api')
DATA_DIR     = os.path.join(API_DIR     , 'data')
IMG_DIR      = os.path.join(DATA_DIR    , 'img')
#GALLERY_DIR  = os.path.join(DST_ROOT_DIR, 'gallery')
#TARGETS_DIR  = os.path.join(DST_ROOT_DIR, 'targets')
RDB_DIR      = os.path.join(DATA_DIR    , 'database')
SVD_MDL_DIR  = os.path.join(API_DIR     , 'saved_models')
SVD_VRF_DIR  = os.path.join(SVD_MDL_DIR , 'verifiers')

# List with all path variables' "names"
directory_list_names = ['api root', 'data dir', 'img dir', 'rep. database',
                        'saved models', 'saved verifiers']

# Tuple with all verifier names
# verifier_names = ('VGG-Face', 'Facenet', 'Facenet512', 'OpenFace',
#                   'DeepFace', 'DeepID' , 'ArcFace')
verifier_names = ('ArcFace', '') # to avoid long startups during developing
                                 # (dont forget to include '' to avoid a bug)

# Other variables
models      = {}             # stores all face verifier models
rep_db      = RepDatabase()  # stores representation database
db_changed  = False          # indicates whether database has been modified (and should be saved)

# SQLAlchemy global variables

SQLITE_DB = 'loki.sqlite'    # SQLite storage name
SQLITE_PATH = os.path.join(RDB_DIR, SQLITE_DB)
sqla_engine = None           # SQLAlchemy engine object