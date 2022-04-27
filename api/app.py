# ==============================================================================
#                                 APP METHODS
# ==============================================================================
import os
import pickle
import api.global_variables   as glb

from fastapi                  import FastAPI
from IFR.api                  import load_database, save_built_model,\
                                    init_load_detectors, init_load_verifiers,\
                                    save_built_detectors, save_built_verifiers
from IFR.functions            import ensure_dirs_exist
from api.routers.recognition  import fr_router
from fastapi.middleware.cors  import CORSMiddleware

# ______________________________________________________________________________
#                               APP INITIALIZATION
# ------------------------------------------------------------------------------

app     = FastAPI(name='Face Recognition API')
origins = ['http://localhost:8080']

app.add_middleware(
    CORSMiddleware,
    allow_origins     = origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(fr_router, prefix="/fr", tags=["Face Recognition"])


# ______________________________________________________________________________
#                                   APP METHODS
# ------------------------------------------------------------------------------

@app.on_event("startup")
async def initialization():
    print('\n ======== Starting initialization process ======== \n')

    # Directories & paths initialization
    print('  -> Directory creation:')
    directory_list = [glb.API_DIR, glb.DATA_DIR, glb.IMG_DIR, glb.RDB_DIR,
                      glb.SVD_MDL_DIR, glb.SVD_VRF_DIR, glb.SVD_DTC_DIR]
    ensure_dirs_exist(directory_list, verbose=True)
    print('')

    # Tries to load a database if it exists. If not, create a new one.
    print('  -> Loading / creating database:')
    glb.sqla_engine = load_database(glb.SQLITE_DB_FP)
    print('')
    
    # Loads (or creates) all face verifiers
    print('  -> Loading / creating face verifiers:')
    glb.models = init_load_verifiers(glb.verifier_names, glb.SVD_VRF_DIR)

    # Loads (or creates) all face detectors
    print('  -> Loading / creating face detectors:')
    glb.models = init_load_detectors(glb.detector_names, glb.SVD_VRF_DIR,
                                     models=glb.models)

    print('\n -------- End of initialization process -------- \n')

# ------------------------------------------------------------------------------

@app.on_event("shutdown")
async def finish_processes():
    print('\n ======== Performing finishing processes ======== \n')

    # Saves the modified database if flag is True
    if glb.db_changed:
        print('  -> Database has changed: saving database.\n')
        db_fp = os.path.join(glb.RDB_DIR, 'rep_database.pickle')

        with open(db_fp, 'wb') as handle:
            pickle.dump(glb.rep_db, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Updates the database changed flag
        glb.db_changed = False
    else:
        print('  -> Database is unchanged: skipping save.\n')

    # Saves (built) face detectors (if needed)
    print('  -> Saving face detectors (if needed):')
    save_built_detectors(glb.detector_names, glb.SVD_DTC_DIR, overwrite=False,
                         verbose=True)

    # Saves (built) face verifiers (if needed)
    print('  -> Saving face verifiers (if needed):')
    save_built_verifiers(glb.verifier_names, glb.SVD_VRF_DIR, overwrite=False,
                         verbose=False)

    print('\n -------- Exitting program: goodbye! -------- \n')

# ------------------------------------------------------------------------------
