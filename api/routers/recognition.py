# ==============================================================================
#                             API CLASSES
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy            as np

from typing             import List
from fastapi            import APIRouter, UploadFile
from deepface           import DeepFace
from ..api_functions    import build_face_verifier, verify_faces,\
                               calculate_similarity
from ..api_classes      import DistanceMetrics, GetRepresentationParams,\
                               FaceVerifierParams
from .detection         import fd_router, FaceDetectorOptions

# --------------------------- ROUTER INITIALIZATION ----------------------------
fr_router = APIRouter()
fr_router.face_verifier = None
fr_router.face_verifier_name = None

# -------------------------------- API METHODS ---------------------------------
@fr_router.post("/verify")
async def face_verify(tgt_file: UploadFile, ref_file: UploadFile,
                      params: FaceVerifierParams):
    """
    API ENDPOINT: verify_faces
        Used to verify if the person in image 1 (target) matches that of image 2
        (reference). Returns a JSON message with the 'verified' status (i.e.
        matches [True] or not [False]), distance between the representation of
        the two images, the threshold for acceptable similarity, the face
        verifier model, face detector model and the distance metric used to
        calculate the distance.

        Note: Please ensure that both images only contain a single person!
    """
    # print('', '-' * 79, f'Model name: {params.model_name}',
    #       f'Distance metric: {params.distance_metric}',
    #       f'Enforce detection: {params.enforce_detection}',
    #       f'Detector backend: {params.detector_backend}',
    #       f'Align: {params.align}',
    #       f'Normalization: {params.normalization}',
    #       f'Threshold: {params.threshold}', '-' * 79, '', sep='\n')
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    tgt_data = np.fromfile(tgt_file.file, dtype=np.uint8)
    tgt_img  = cv2.imdecode(tgt_data, cv2.IMREAD_UNCHANGED)

    ref_data = np.fromfile(ref_file.file, dtype=np.uint8)
    ref_img  = cv2.imdecode(ref_data, cv2.IMREAD_UNCHANGED)

    print(f'\nFace verifier name: {fr_router.face_verifier_name}',
          f'Face verifier: {fr_router.face_verifier}', sep='\n')

    # Builds the face verifier model
    fr_router.face_verifier_name, metric_name, fr_router.face_verifier = \
                build_face_verifier(model_name=params.model_name, 
                                    model=fr_router.face_verifier,
                                    distance_metric=params.distance_metric)

    # Unnests single element list into element. Does the same for the dictionary
    # of models unless the model name is 'Ensemble' (as Ensemble models consists
    # of 4 models)
    fr_router.face_verifier_name = fr_router.face_verifier_name[0]
    metric_name                  = metric_name[0]

    if fr_router.face_verifier_name != "Ensemble":
        fr_router.face_verifier = \
            fr_router.face_verifier[fr_router.face_verifier_name]

    # Check if a face detection model has been used before, if not use a default
    # option
    if fd_router.face_detector_name == None:
        use_backend = FaceDetectorOptions.OPENCV
    else:
        use_backend = fd_router.face_detector_name

    # print('', f'Face verifier name: {fr_router.face_verifier_name}',
    #       f'Metric name: {metric_name}',
    #       f'Face verifier: {fr_router.face_verifier}',
    #       f'Face detector: {use_backend}', '', sep='\n')

    # Verifies both faces
    response = verify_faces(tgt_img, img2_path=ref_img,
                    model_name = fr_router.face_verifier_name,
                    distance_metric = metric_name,
                    model = fr_router.face_verifier,
                    enforce_detection = params.enforce_detection,
                    detector_backend = use_backend,
                    align = params.align,
                    prog_bar = False,
                    normalization = params.normalization,
                    threshold = params.threshold)

    #print('\n', '-' * 80, '\n', response, '\n', '-' * 80, '\n')

    return response

# ------------------------------------------------------------------------------
@fr_router.post("/represent")
async def get_representation(myfile: UploadFile,
                             params: GetRepresentationParams):
    """
    API ENDPOINT: verify_faces
        Used to get the vector representation of a person in an image. Make sure
        the image provided contains only ONE person, otherwise, the
        representation of the first person detected will be returned. Returns a
        JSON message with ...
    """
    # print('', '-' * 79, f'Model name: {params.model_name}',
    #       f'Distance metric: {params.distance_metric}',
    #       f'Enforce detection: {params.enforce_detection}',
    #       f'Detector backend: {params.detector_backend}',
    #       f'Align: {params.align}',
    #       f'Normalization: {params.normalization}',
    #       f'Threshold: {params.threshold}', '-' * 79, '', sep='\n')
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    data = np.fromfile(myfile.file, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Builds the face verifier model
    fr_router.face_verifier_name, _junk, fr_router.face_verifier = \
                build_face_verifier(model_name=params.model_name, 
                                    model=fr_router.face_verifier,
                                    distance_metric='cosine')

    # Unnests single element list into element. Does the same for the dictionary
    # of models unless the model name is 'Ensemble' (as Ensemble models consists
    # of 4 models)
    fr_router.face_verifier_name = fr_router.face_verifier_name[0]

    if fr_router.face_verifier_name != "Ensemble":
        fr_router.face_verifier = \
            fr_router.face_verifier[fr_router.face_verifier_name]

    # Check if a face detection model has been used before, if not use a default
    # option
    if fd_router.face_detector_name == None:
        use_backend = FaceDetectorOptions.OPENCV
    else:
        use_backend = fd_router.face_detector_name

    representation = DeepFace.represent(img, model_name = params.model_name,
                                model = fr_router.face_verifier,
                                enforce_detection = params.enforce_detection,
                                detector_backend = use_backend,
                                align = params.align,
                                normalization = params.normalization)

    return {'representation':representation}

# ------------------------------------------------------------------------------
@fr_router.post("/distance")
async def calculate_distance(rep_1: List[float], rep_2: List[float],
                             metrics: List[DistanceMetrics] = \
                                            [DistanceMetrics.COSINE,
                                             DistanceMetrics.EUCLIDEAN,
                                             DistanceMetrics.EUCLIDEAN_L2]):
    # Converts to numpy arrays
    rep_1 = np.array(rep_1)
    rep_2 = np.array(rep_2)

    # Calculates similarity / distance
    distances = calculate_similarity(rep_1, rep_2, metrics=metrics)
    return distances

# ------------------------------------------------------------------------------
@fr_router.post("/analyze/emotion")
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
@fr_router.post("/analyze/age")
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
@fr_router.post("/analyze/gender")
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
@fr_router.post("/analyze/race")
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
@fr_router.post("/analyze/all")
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

