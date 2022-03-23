# ==============================================================================
#                           APP-WIDE (GLOBAL) VARIABLES
# ==============================================================================

import os

# Path variables
API_DIR      = os.path.join(os.path.dirname(os.path.realpath("__file__")),'api')
DST_ROOT_DIR = os.path.join(API_DIR     , 'data')
RAW_DIR      = os.path.join(DST_ROOT_DIR, 'raw')
#GALLERY_DIR  = os.path.join(DST_ROOT_DIR, 'gallery')
TARGETS_DIR  = os.path.join(DST_ROOT_DIR, 'targets')
RDB_DIR      = os.path.join(DST_ROOT_DIR, 'database')
SVD_MDL_DIR  = os.path.join(API_DIR     , 'saved_models')
SVD_VRF_DIR  = os.path.join(SVD_MDL_DIR , 'verifiers')

# List with all path variables' "names"
directory_list_names = ['api root', 'dataset root', 'raw', 'targets',
                        'rep. database', 's. models', 's. verifiers']

# Other variables
models      = []    # stores all face verifier models
rep_db      = []    # stores representation database
db_changed  = False # indicates whether database has been modified (and should be saved)
