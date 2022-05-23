# ==============================================================================
#                                 APP METHODS
# ==============================================================================
import os
import pickle
import api.global_variables   as glb

from fastapi                  import FastAPI
from fastapi.middleware.cors  import CORSMiddleware
from IFR.api                  import load_database, init_load_detectors,\
                                    init_load_verifiers, save_built_detectors,\
                                    save_built_verifiers, start_session
from IFR.functions            import ensure_dirs_exist,\
                                    remove_img_file_duplicates
from api.routers.recognition  import fr_router

# ______________________________________________________________________________
#                               APP INITIALIZATION
# ------------------------------------------------------------------------------

app     = FastAPI(name='Face Recognition API')
origins = ['http://localhost:8000, http://localhost:8080']

app.add_middleware(
    CORSMiddleware,
    allow_origins     = origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(fr_router, prefix="/api", tags=["Face Recognition"])


# ______________________________________________________________________________
#                                   APP METHODS
# ------------------------------------------------------------------------------

@app.on_event("startup")
async def initialization():
    if glb.DEBUG:
        print('\n ======== Starting initialization process ======== \n')

    # Directories & paths initialization
    if glb.DEBUG:
        print('  -> Directory creation:')
    directory_list = [glb.API_DIR, glb.DATA_DIR, glb.IMG_DIR, glb.RDB_DIR,
                      glb.SVD_MDL_DIR, glb.SVD_VRF_DIR, glb.SVD_DTC_DIR]
    ensure_dirs_exist(directory_list, verbose=True)
    if glb.DEBUG:
        print('')

    # Tries to load a database if it exists. If not, create a new one.
    if glb.DEBUG:
        print('  -> Loading / creating database:', end='')
    glb.sqla_engine = load_database(glb.SQLITE_DB_FP)
    if glb.DEBUG:
        print('success!\n')

    # Loads (or creates) the session. Also commits once to create table
    # definitions if required.
    if glb.DEBUG:
        print('  -> Loading / creating session:', end='')
    glb.sqla_session = start_session(glb.sqla_engine)
    glb.sqla_session.commit()                   # create table definitions
    if glb.DEBUG:
        print('success!\n')
    
    # Loads (or creates) all face verifiers
    if glb.DEBUG:
        print('  -> Loading / creating face verifiers:')
    glb.models = init_load_verifiers(glb.verifier_names, glb.SVD_VRF_DIR)
    if glb.DEBUG:
        print('')

    # Loads (or creates) all face detectors
    if glb.DEBUG:
        print('  -> Loading / creating face detectors:')
    glb.models = init_load_detectors(glb.detector_names, glb.SVD_VRF_DIR,
                                     models=glb.models)

    # Ensures no duplicate image files exists in the server's image directory
    if glb.DEBUG:
        print('  -> Ensuring no duplicate image files exist:')
    dup_file_names = remove_img_file_duplicates(glb.IMG_DIR, dont_delete=False)
    if glb.DEBUG:
        for name in dup_file_names:
            print(f' > Deleted duplicate file: {name}')
        print('')

    if glb.DEBUG:
        print('\n -------- End of initialization process -------- \n')

# ------------------------------------------------------------------------------

@app.on_event("shutdown")
async def finish_processes():
    if glb.DEBUG:
        print('\n ======== Performing finishing processes ======== \n')

    if not glb.skip_model_save: # this is just to speed up my (Rodrigo's)
                                # development but will be removed later in
                                # production
        # Saves (built) face detectors (if needed)
        if glb.DEBUG:
            print('  -> Saving face detectors (if needed):')
        save_built_detectors(glb.detector_names, glb.SVD_DTC_DIR,
                             overwrite=False, verbose=True)

        # Saves (built) face verifiers (if needed)
        if glb.DEBUG:
            print('  -> Saving face verifiers (if needed):')
        save_built_verifiers(glb.verifier_names, glb.SVD_VRF_DIR,
                             overwrite=False, verbose=False)
    
    if glb.DEBUG:
        print('\n -------- Exitting program: goodbye! -------- \n')

# ------------------------------------------------------------------------------
