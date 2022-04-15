# ==============================================================================
#                                 APP METHODS
# ==============================================================================
import os
import pickle
import api.global_variables         as glb

from fastapi                        import FastAPI
from IFR.api                        import load_representation_db,\
                                        load_face_verifier, save_face_verifier
from IFR.functions                  import create_dir
from api.routers.recognition        import fr_router

from deepface.DeepFace              import build_model    as build_verifier

# ______________________________________________________________________________
#                               APP INITIALIZATION
# ------------------------------------------------------------------------------

app = FastAPI(name='Face Recognition API')
app.include_router(fr_router , prefix="/fr" , tags=["Face Recognition"])

# ______________________________________________________________________________
#                                   APP METHODS
# ------------------------------------------------------------------------------

@app.on_event("startup")
async def initialization():
    print('\n ======== Starting initialization process ======== \n')


    # Directories & paths initialization
    print('  -> Directory creation:')
    directory_list = [glb.API_DIR, glb.DATA_DIR, glb.IMG_DIR, glb.RDB_DIR,
                      glb.SVD_MDL_DIR, glb.SVD_VRF_DIR]
    
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
    

    # Loads (or creates) all face verifiers
    print('  -> Loading / creating face verifiers:')

    for verifier_name in glb.verifier_names:
        # Quick fix to avoid problems when len(glb.verifier_names) == 1. In this
        # case, FOR loops over each letter in the string instead of considering
        # the entire string as one thing.
        if verifier_name == '':
            continue

        # First, try loading (opening) the model
        model = load_face_verifier(verifier_name + '.pickle', glb.SVD_VRF_DIR,
                                   verbose=True)

        # If successful, save the model in a dictionary
        if not isinstance(model, list):
            glb.models[verifier_name] = model

        # Otherwise, build the model from scratch
        else:
            print(f'[build_verifier] Building {verifier_name}: ', end='')
            try:
                glb.models[verifier_name] = build_verifier(verifier_name)
                print('success!\n')

            except Exception as excpt:
                print(f'failed! Reason: {excpt}\n')


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


    # Saves (built) face verifiers (if needed)
    print('  -> Saving face verifiers (if needed):')
    for verifier_name in glb.verifier_names:
        # Quick fix to avoid problems when len(glb.verifier_names) == 1. In this
        # case, FOR loops over each letter in the string instead of considering
        # the entire string as one thing.
        if verifier_name == '':
            continue

        # Saving face verifiers
        save_face_verifier(verifier_name, glb.models[verifier_name],
                           glb.SVD_VRF_DIR, overwrite=False, verbose=True)


    print('\n -------- Exitting program: goodbye! -------- \n')

# ------------------------------------------------------------------------------
