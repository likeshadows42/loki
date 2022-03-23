# ==============================================================================
#                                 APP METHODS
# ==============================================================================
import os
import pickle

from fastapi                        import FastAPI
from api_functions                 import API_DIR, DST_ROOT_DIR, RAW_DIR,\
                                           TARGETS_DIR, RDB_DIR, SVD_MDL_DIR,\
                                           SVD_VRF_DIR, load_representation_db,\
                                           create_dir, load_face_verifier

import global_variables as glb

from router_detection          import fd_router
from router_verification       import fv_router
from router_recognition        import fr_router
from router_attribute_analysis import aa_router

# ______________________________________________________________________________
#                               APP INITIALIZATION
# ------------------------------------------------------------------------------

app = FastAPI()

app.include_router(fd_router, prefix="/fd", tags=["Face Detection"])
app.include_router(fv_router, prefix="/fv", tags=["Face Verification"])
app.include_router(fr_router, prefix="/fr", tags=["Face Recognition"])
app.include_router(aa_router, prefix="/aa", tags=["Face Attribute Analysis"])

directory_list       = [API_DIR, DST_ROOT_DIR, RAW_DIR, TARGETS_DIR,
                        RDB_DIR, SVD_MDL_DIR, SVD_VRF_DIR]
directory_list_names = ['api root', 'dataset root', 'raw', 'targets',
                        'rep. database', 's. models', 's. verifiers']

# ______________________________________________________________________________
#                                   APP METHODS
# ------------------------------------------------------------------------------

@app.on_event("startup")
async def initialization():
    print('\n ======== Starting initialization process ======== \n')
    # print('  -> Path names:')
    # for name, fp in zip(directory_list_names, directory_list):
    #     print(f'Directory {name}'.ljust(25), f': {fp}', sep='')
    # print('')

    # # REMOVE AFTER DEBUGGING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print('Exited by debug. Good bye.')
    # return None

    # Directories & paths initialization
    print('  -> Directory creation:')
    for directory in directory_list:
        print(f'Creating {directory} directory: ', end='')
    
        if create_dir(directory):
            print('directory exists. Continuing...')
        else:
            print('success.')
    print('')

    # Tries to load a database if it exists. If not, create a new one.
    print('  -> Loading / creating database:')
    if not os.path.isfile(os.path.join(RDB_DIR, 'rep_database.pickle')):
        glb.db_changed = True
    glb.rep_db = load_representation_db(os.path.join(RDB_DIR,
                                    'rep_database.pickle'), verbose=True)
    print('')
    
    # Checks if face verifier folder exists
    print('  -> Loading / creating face verifiers:')
    glb.models = load_face_verifier(['ArcFace'],
                                    save_dir=SVD_VRF_DIR,
                                    show_prog_bar=False, verbose=True)
    print('\n -------- End of initialization process -------- \n')

# ------------------------------------------------------------------------------

@app.on_event("shutdown")
async def finish_processes():
    print('\n ======== Performing finishing processes ======== \n')

    # Saves the modified database if flag is True
    if glb.db_changed:
        print('  -> Database has changed: saving database.')
        db_fp = os.path.join(RDB_DIR, 'rep_database.pickle')

        with open(db_fp, 'wb') as handle:
            pickle.dump(glb.rep_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('  -> Database is unchanged: skipping save.')



    print('\n -------- Exitting program: good bye! -------- \n')

# ------------------------------------------------------------------------------
