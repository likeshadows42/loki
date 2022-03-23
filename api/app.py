# ==============================================================================
#                                 APP METHODS
# ==============================================================================
import os
import pickle

from fastapi                        import FastAPI     
from .api_functions                 import load_representation_db,\
                                           load_face_verifier, create_dir
from .                              import global_variables as glb

from api.routers.detection          import fd_router
from api.routers.verification       import fv_router
from api.routers.recognition        import fr_router
from api.routers.attribute_analysis import aa_router

# ______________________________________________________________________________
#                               APP INITIALIZATION
# ------------------------------------------------------------------------------

app = FastAPI()

app.include_router(fd_router, prefix="/fd", tags=["Face Detection"])
app.include_router(fv_router, prefix="/fv", tags=["Face Verification"])
app.include_router(fr_router, prefix="/fr", tags=["Face Recognition"])
app.include_router(aa_router, prefix="/aa", tags=["Face Attribute Analysis"])

# ______________________________________________________________________________
#                                   APP METHODS
# ------------------------------------------------------------------------------

@app.on_event("startup")
async def initialization():
    print('\n ======== Starting initialization process ======== \n')

    # Directories & paths initialization
    print('  -> Directory creation:')
    directory_list = [glb.API_DIR    , glb.DST_ROOT_DIR, glb.RAW_DIR,
                      glb.TARGETS_DIR, glb.RDB_DIR     , glb.SVD_MDL_DIR,
                      glb.SVD_VRF_DIR]
    
    for directory in directory_list:
        print(f'Creating {directory} directory: ', end='')
    
        if create_dir(directory):
            print('directory exists. Continuing...')
        else:
            print('success.')
    print('')

    # Tries to load a database if it exists. If not, create a new one.
    print('  -> Loading / creating database:')
    if not os.path.isfile(os.path.join(glb.RDB_DIR, 'rep_database.pickle')):
        glb.db_changed = True
    glb.rep_db = load_representation_db(os.path.join(glb.RDB_DIR,
                                    'rep_database.pickle'), verbose=True)
    print('')
    
    # Checks if face verifier folder exists
    print('  -> Loading / creating face verifiers:')
    glb.models = load_face_verifier(['ArcFace'],
                                    save_dir=glb.SVD_VRF_DIR,
                                    show_prog_bar=False, verbose=True)
    print('\n -------- End of initialization process -------- \n')

# ------------------------------------------------------------------------------

@app.on_event("shutdown")
async def finish_processes():
    print('\n ======== Performing finishing processes ======== \n')

    # Saves the modified database if flag is True
    if glb.db_changed:
        print('  -> Database has changed: saving database.')
        db_fp = os.path.join(glb.RDB_DIR, 'rep_database.pickle')

        with open(db_fp, 'wb') as handle:
            pickle.dump(glb.rep_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('  -> Database is unchanged: skipping save.')



    print('\n -------- Exitting program: good bye! -------- \n')

# ------------------------------------------------------------------------------
