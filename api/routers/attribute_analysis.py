# ==============================================================================
#                    ATTRIBUTE ANALYSIS-RELATED API METHODS
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy            as np

from fastapi            import APIRouter, UploadFile
from deepface           import DeepFace
from ..api_classes      import FaceDetectorOptions
from .recognition       import fr_router

# --------------------------- ROUTER INITIALIZATION ----------------------------
aa_router = APIRouter()

# -------------------------------- API METHODS ---------------------------------
@aa_router.post("/emotion")
async def analyze_emotion(myfile: UploadFile, enforce_detection: bool = True, 
            detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV):
    """
    API ENDPOINT: analyze_emotion
        Used to determine the person's emotion based on the image. Returns a 
        dictionary with the probability of each emotion, dominant emotion (most
        likely emotion) and face region.
    """
    # Creates a dictionary with the current model as function expects a
    # dictionary
    if fr_router.face_verifier != None and fr_router.face_verifier_name != None:
        models = {fr_router.face_verifier_name:fr_router.face_verifier}
    else:
        models = None
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    data = np.fromfile(myfile.file, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Analyzes the emotion attribute from the image
    response_obj = DeepFace.analyze(img,
                    actions = ('emotion',),
                    models = models,
                    enforce_detection = enforce_detection,
                    detector_backend = detector_backend,
                    prog_bar = False)

    return {'response_obj':response_obj}

# ------------------------------------------------------------------------------
@aa_router.post("/age")
async def analyze_age(myfile: UploadFile, enforce_detection: bool = True, 
            detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV):
    """
    API ENDPOINT: analyze_age
        Used to determine the person's age based on the image. Returns a 
        dictionary with age, dominant emotion and region.
    """
    # Creates a dictionary with the current model as function expects a
    # dictionary
    if fr_router.face_verifier != None and fr_router.face_verifier_name != None:
        models = {fr_router.face_verifier_name:fr_router.face_verifier}
    else:
        models = None
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    data = np.fromfile(myfile.file, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Analyzes the emotion attribute from the image
    response_obj = DeepFace.analyze(img,
                    actions = ('age',),
                    models = models,
                    enforce_detection = enforce_detection,
                    detector_backend = detector_backend,
                    prog_bar = False)

    return {'response_obj':response_obj}

# ------------------------------------------------------------------------------
@aa_router.post("/gender")
async def analyze_gender(myfile: UploadFile, enforce_detection: bool = True, 
            detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV):
    """
    API ENDPOINT: analyze_gender
        DISCLAIMER: The word 'gender' is used incorretly as only male / female,
        that is, sexs, are considered.

        Used to determine the person's gender (actually sex) based on the image.
        Returns the person's sex and face region.
    """
    # Creates a dictionary with the current model as function expects a
    # dictionary
    if fr_router.face_verifier != None and fr_router.face_verifier_name != None:
        models = {fr_router.face_verifier_name:fr_router.face_verifier}
    else:
        models = None
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    data = np.fromfile(myfile.file, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Analyzes the emotion attribute from the image
    response_obj = DeepFace.analyze(img,
                    actions = ('gender',),
                    models = models,
                    enforce_detection = enforce_detection,
                    detector_backend = detector_backend,
                    prog_bar = False)

    return {'response_obj':response_obj}

# ------------------------------------------------------------------------------
@aa_router.post("/race")
async def analyze_race(myfile: UploadFile, enforce_detection: bool = True, 
            detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV):
    """
    API ENDPOINT: analyze_race
        Used to determine the person's race based on the image. Returns a 
        dictionary with the probability of each race, dominant (most likely)
        race and face region.
    """
    # Creates a dictionary with the current model as function expects a
    # dictionary
    if fr_router.face_verifier != None and fr_router.face_verifier_name != None:
        models = {fr_router.face_verifier_name:fr_router.face_verifier}
    else:
        models = None
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    data = np.fromfile(myfile.file, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Analyzes the emotion attribute from the image
    response_obj = DeepFace.analyze(img,
                    actions = ('race',),
                    models = models,
                    enforce_detection = enforce_detection,
                    detector_backend = detector_backend,
                    prog_bar = False)

    return {'response_obj':response_obj}

# ------------------------------------------------------------------------------
@aa_router.post("/all")
async def analyze_all(myfile: UploadFile, enforce_detection: bool = True, 
            detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV):
    """
    API ENDPOINT: analyze_all
        Used to determine the person's emotion, age, race and gender based on
        the image. Returns a dictionary with the probability of each emotion, 
        and race, dominant (most likely) emotion and race, gender (sex), age
        and face region.
    """
    # Creates a dictionary with the current model as function expects a
    # dictionary
    if fr_router.face_verifier != None and fr_router.face_verifier_name != None:
        models = {fr_router.face_verifier_name:fr_router.face_verifier}
    else:
        models = None
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    data = np.fromfile(myfile.file, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Analyzes the emotion attribute from the image
    response_obj = DeepFace.analyze(img,
                    actions = ('emotion', 'age', 'gender', 'race'),
                    models = models,
                    enforce_detection = enforce_detection,
                    detector_backend = detector_backend,
                    prog_bar = False)

    return {'response_obj':response_obj}
