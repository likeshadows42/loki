# ==============================================================================
#                               APP ENDPOINT TESTS
# ==============================================================================

# IMPLEMENTATION NOTE: The test functions start with hardcodded parameters to
# build the POST requests. This is a workaround to be able to change default
# values. They are declared within the function's body to enable an easier
# changed of values if needed.

import os

from .                      import global_variables   as glb
from .app                   import app
from IFR.classes            import default_detector, default_verifier,\
                                default_metric, default_normalization,\
                                default_image_save_type, default_align,\
                                default_enforce_detection, default_threshold,\
                                default_tags, default_uids, default_verbose,\
                                default_msg_detail, default_msg_output,\
                                default_property
from IFR.functions          import create_dir, load_representation_db,\
                                load_face_verifier
from deepface.DeepFace      import build_model       as build_verifier
from fastapi.testclient     import TestClient

# ______________________________________________________________________________
#                           TEST CLIENT INITIALIZATION
# ------------------------------------------------------------------------------

client = TestClient(app)

# Editting local paths because here they end up with /api/api
glb.API_DIR      = os.path.dirname(os.path.realpath("__file__"))
glb.DATA_DIR     = os.path.join(glb.API_DIR    , 'data')
glb.IMG_DIR      = os.path.join(glb.DATA_DIR   , 'img')
glb.RDB_DIR      = os.path.join(glb.DATA_DIR   , 'database')
glb.SVD_MDL_DIR  = os.path.join(glb.API_DIR    , 'saved_models')
glb.SVD_VRF_DIR  = os.path.join(glb.SVD_MDL_DIR, 'verifiers')

# Paths to files used for testing
test_zip_file    = '/home/rpessoa/projects/loki/api/test_zip.zip'

# For use during development - remove me later
skip_server_reset_test = True

# ______________________________________________________________________________
#                           EMULATE SERVER ENVIRONMENT
# ------------------------------------------------------------------------------

print('\n ======== Starting initialization process ======== \n')

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

# ______________________________________________________________________________
#                                   APP TESTS
# ------------------------------------------------------------------------------

def test_inspect_globals():
    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.get("/fr/debug/inspect_globals")
    out = dict(r.json())

    # Checks status code first
    assert r.status_code == 200

    # Ensures directories (paths) and directory names contain at least 1 entry
    # and have the same length
    assert len(out['directories']) > 0 and len(out['dir_names']) > 0\
        and len(out['directories']) == len(out['dir_names'])

    # Ensures that the 'models' object is either a dictionary or list, and that
    # it contains at least 1 model
    assert type(out['models']) == list or type(out['models']) == dict\
        and len(out['models']) > 0

    # Ensures that the loaded database exists by checking that it has 0 or more
    # Representations
    assert type(out['num_reps']) == int and out['num_reps'] >= 0

    # Ensures that the 'database has been modified' flag exists (and is a
    # boolean)
    assert type(out['db_changed']) == bool

# ------------------------------------------------------------------------------

if not skip_server_reset_test:  # this test takes long so you can skip it
    def test_reset_server():
        # Gets the response object and converts the JSON payload to a dictionary
        r   = client.post("/fr/debug/server_reset")
        out = dict(r.json())

        # Checks status code first
        assert r.status_code == 200

        # Ensures the response message states the server has been restarted
        assert out['message'].lower() == "Server has been restarted".lower()

# ------------------------------------------------------------------------------
# this is not a "real" endpoint. it will be remove, it is just being used to
# test why default values were not being change.

def test_default_values_change():
    # Test parameters - can be changed despite being hardcodded
    tst_value1 =  1000
    tst_value2 = -9999

    # Create client post request
    c_post  = r"/fr/debug/default_values_change"\
            + f"?value1={tst_value1}&value2={tst_value2}"
    
    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())
    out = dict(r.json())

    # Ensures the values are different from the default values
    assert out['value1'] !=  10
    assert out['value2'] != -33

# ------------------------------------------------------------------------------

def test_edit_default_directories_BAD_DIRECTORIES():
    # Test parameters - can be changed despite being hardcodded
    tst_bad_img_dir = r"%2Fsome%2Frandom%2Fbad%2Fdirectory%2Fpath%2Ffor%2Fimg"
    tst_bad_rdb_dir = r"%2Fsome%2Frandom%2Fbad%2Fdirectory%2Fpath%2Ffor%2Fdatabase"

    # Create client post request
    c_post  = r"/fr/utility/edit_default_directories"\
            + f"?img_dir={tst_bad_img_dir}&rdb_dir={tst_bad_rdb_dir}"
    
    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())
    out = dict(r.json())

    # Checks status code first
    assert r.status_code == 200

    # Ensures the directories have NOT changed (they havent as they are bad
    # directories)
    assert out['status'] == True

def test_edit_default_directories_VALID_DIRECTORIES():
    # Test parameters - can be changed despite being hardcodded
    tst_valid_dir = r"%2Fhome%2Frpessoa%2Fprojects%2Floki%2Fapi%2F"

    # Create client post request
    c_post  = r"/fr/utility/edit_default_directories"\
            + f"?img_dir={tst_valid_dir}&rdb_dir={tst_valid_dir}"
    
    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())
    out = dict(r.json())

    # Checks status code first
    assert r.status_code == 200

    # Ensures the directories HAVE changed (they should as they are valid
    # directories)
    assert out['status'] == False

# ------------------------------------------------------------------------------

def test_get_property_from_database_BAD_PROPERTY():
    # Test parameters - can be changed despite being hardcodded
    tst_bad_property   = "BAD_PROPERTY"
    tst_do_sort        = False
    tst_suppress_error = True

    # Create client post request
    c_post  = r"/fr/utility/get_property_from_database"\
            + f"?propty={tst_bad_property}"\
            + f"&do_sort={tst_do_sort}"\
            + f"&suppress_error={tst_suppress_error}"
    
    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)

    # Checks status code first
    assert r.status_code == 422

def test_get_property_from_database_IS_SORTED():
    # Test parameters - can be changed despite being hardcodded
    tst_good_property   = "name_tag"
    tst_do_sort        = True
    tst_suppress_error = True

    # Create client post request
    c_post  = r"/fr/utility/get_property_from_database"\
            + f"?propty={tst_good_property}"\
            + f"&do_sort={tst_do_sort}"\
            + f"&suppress_error={tst_suppress_error}"
    
    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())  # good for debugging - only prints on error / bad asserts
    out = r.json()

    # Checks status code first
    assert r.status_code == 200

    # Checks if the output is sorted
    assert out == sorted(out)

# ------------------------------------------------------------------------------

def test_view_database():
    # Test parameters - can be changed despite being hardcodded
    tst_amt_detail  = "summary"
    tst_output_type = "structure"

    # Create client post request
    c_post  = r"/fr/utility/view_database"\
            + f"?amt_detail={tst_amt_detail}"\
            + f"&output_type={tst_output_type}"
    
    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())  # good for debugging - only prints on error / bad asserts
    out = r.json()

    # Checks status code first
    assert r.status_code == 200

    # Checks if the output has the same size as the database (it should!)
    assert len(out) == len(glb.rep_db)

# ------------------------------------------------------------------------------

def test_edit_tag_by_uid():
    # Test parameters - can be changed despite being hardcodded
    tst_tgt_uid = "64721bd7-2b08-437f-8682-7afc6f84b63c"
    tst_new_tag = "Angelina%20123"

    # Create client post request
    c_post  = r"/fr/utility/edit_tag_by_uid/"\
            + f"?target_uid={tst_tgt_uid}&new_name_tag={tst_new_tag}"

    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())  # good for debugging - only prints on error / bad asserts
    out = dict(r.json())

    # Checks status code first
    assert r.status_code == 200

    # Asserts that the name tag has been appropriately changed for the correct
    # uid
    assert out['unique_id'] == tst_tgt_uid
    assert out['name_tag']  == tst_new_tag.replace('%20', ' ')

# ------------------------------------------------------------------------------

def test_search_database_by_tag_BAD_TAG():
    # Test parameters - can be changed despite being hardcodded
    target_tag = r'This%20Tag%20Does%20Not%20Exist'

    # Create client post request
    c_post  = r"/fr/utility/search_database_by_tag/"\
            + f"?target_tag={target_tag}"

    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())  # good for debugging - only prints on error / bad asserts

    # Checks status code first
    assert r.status_code == 200

    # Ensures that no Representation was found (i.e. an empty list is returned)
    assert len(r.json()) == 0

def test_search_database_by_tag_GOOD_TAG():
    # Test parameters - can be changed despite being hardcodded
    target_tag = r'Angelina%20Jolie'

    # Create client post request
    c_post  = r"/fr/utility/search_database_by_tag/"\
            + f"?target_tag={target_tag}"

    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())  # good for debugging - only prints on error / bad asserts

    # Checks status code first
    assert r.status_code == 200

    # Ensures that the name tag matches the target tag in each Representation
    # found
    for json_rep in r.json():
        rep = dict(json_rep)
        assert rep['name_tag'] == target_tag.replace('%20', ' ')

# ------------------------------------------------------------------------------

def test_rename_entries_by_tag_BAD_TAG():
    # Test parameters - can be changed despite being hardcodded
    old_tag = r'Nobody_is_called_this'
    new_tag = r'Bobby%20Fischer'

    # Create client post request
    c_post  = r"/fr/utility/rename_entries_by_tag/"\
            + f"?old_tag={old_tag}&new_tag={new_tag}"

    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())  # good for debugging - only prints on error / bad asserts

    # Checks status code first
    assert r.status_code == 200

    # Ensures that no Representation was found (i.e. an empty list is returned)
    assert len(r.json()) == 0

def test_rename_entries_by_tag_GOOD_TAG():
    # Test parameters - can be changed despite being hardcodded
    old_tag = r'Angelina%20Jolie'
    new_tag = r'Mister%20Nobody'

    # Create client post request
    c_post  = r"/fr/utility/rename_entries_by_tag/"\
            + f"?old_tag={old_tag}&new_tag={new_tag}"

    # Gets the response object and converts the JSON payload to a dictionary
    r   = client.post(c_post)
    print(r.json())  # good for debugging - only prints on error / bad asserts

    # Checks status code first
    assert r.status_code == 200

    # Ensures that the name tag matches the new tag in each updated
    # Representation
    for json_rep in r.json():
        rep = dict(json_rep)
        assert rep['name_tag'] == new_tag.replace('%20', ' ')

# ------------------------------------------------------------------------------










# ------------------------------------------------------------------------------

# def test_create_database_from_directory():
#     # Gets the response object and converts the JSON payload to a dictionary
#     r   = client.post("/fr/create_database/from_directory",\
#         json={"detector_name": "retinaface", "verifier_names": ["ArcFace"],\
#               "align": "true", "normalization": "base", "tags": [], "uids": [],\
#               "verbose": "false"}, data={'force_create':True})
#     out = dict(r.json())

#     # Checks status code first
#     assert r.status_code == 200

#     # Ensures that the database created is a list and has a length of 0 or more
#     assert type(glb.rep_db) == list
#     assert out['length']    >= 0

# ------------------------------------------------------------------------------

# def test_create_database_from_zip():
# #    myfile: UploadFile,
# #     params      : CreateDatabaseParams = Depends(),
# #     image_dir   : Optional[str]  = Query(glb.IMG_DIR, description="Full path to directory containing images (string)"),
# #     db_dir      : Optional[str]  = Query(glb.RDB_DIR, description="Full path to directory containing saved database (string)"),
# #     auto_rename : Optional[bool] = Query(True       , description="Flag to force auto renaming of images in the zip file with names that match images already in the image directory (boolean)"),
# #     force_create: Optional[bool] = Query(False      , description="Flag to force database creation even if one already exists, overwritting the old one (boolean)")):

#     # Gets the response object and converts the JSON payload to a dictionary
#     r   = client.post("/fr/create_database/from_zip",
#         files={'upload_file': open(test_zip_file, 'rb')},
#         data ={'detector_name':default_detector,
#                'verifier_names':[default_verifier], 'align':default_align,
#                'normalization':default_normalization, 'tags':default_tags,
#                'uids':default_uids, 'verbose':default_verbose,
#                'force_create':True})

#     out = dict(r.json())
#     print(out)

#     # Checks status code first
#     assert r.status_code == 200

#     assert False, "debugging - remove later"


# ------------------------------------------------------------------------------
