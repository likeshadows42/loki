# ==============================================================================
#                         DEEPFACE-RELATED API METHODS
# ==============================================================================
# These are API methods that rely on ready-made deepface functions.

import os
import cv2
import numpy                 as np
import api.global_variables  as glb

from PIL                     import Image
from typing                  import List

from fastapi                 import APIRouter, UploadFile
from fastapi.responses       import FileResponse

from deepface                import DeepFace
from deepface.detectors      import FaceDetector

from IFR.functions           import build_face_verifier, verify_faces,\
                                    calculate_similarity
from IFR.functions           import detect_faces as find_faces
from IFR.classes             import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_dir = glb.DATA_DIR
img_dir  = glb.IMG_DIR
rdb_dir  = glb.RDB_DIR


# ______________________________________________________________________________
#                             ROUTER INITIALIZATION
# ------------------------------------------------------------------------------

df_router                    = APIRouter()
df_router.face_detector      = None
df_router.face_detector_name = None


# ______________________________________________________________________________
#                     ATTRIBUTE ANALYSIS-RELATED API METHODS
# ------------------------------------------------------------------------------

@df_router.post("/attribute_analysis/emotion")
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
    if df_router.face_verifier != None and df_router.face_verifier_name != None:
        models = {df_router.face_verifier_name:df_router.face_verifier}
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

@df_router.post("/attribute_analysis/age")
async def analyze_age(myfile: UploadFile, enforce_detection: bool = True, 
            detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV):
    """
    API ENDPOINT: analyze_age
        Used to determine the person's age based on the image. Returns a 
        dictionary with age, dominant emotion and region.
    """
    # Creates a dictionary with the current model as function expects a
    # dictionary
    if df_router.face_verifier != None and df_router.face_verifier_name != None:
        models = {df_router.face_verifier_name:df_router.face_verifier}
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

@df_router.post("/attribute_analysis/gender")
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
    if df_router.face_verifier != None and df_router.face_verifier_name != None:
        models = {df_router.face_verifier_name:df_router.face_verifier}
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

@df_router.post("/attribute_analysis/race")
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
    if df_router.face_verifier != None and df_router.face_verifier_name != None:
        models = {df_router.face_verifier_name:df_router.face_verifier}
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

@df_router.post("/attribute_analysis/all")
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
    if df_router.face_verifier != None and df_router.face_verifier_name != None:
        models = {df_router.face_verifier_name:df_router.face_verifier}
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


# ______________________________________________________________________________
#                         DETECTION-RELATED API METHODS 
# ------------------------------------------------------------------------------

@df_router.post("/detection/overlay_faces")
async def overlay_faces(myfile: UploadFile, backend: FaceDetectorOptions,
                             align: bool = False):
    """
    API ENDPOINT: overlay_faces
        Used to detect multiple faces and overlay them on the original image.
        The detected faces are shown with red rectangles. Returns an image.
    """
    
    # Obtains contents of the file & transforms it into an image
    data  = np.fromfile(myfile.file, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Creates a face detector if one does not exist with the same name
    if df_router.face_detector_name != backend:
        df_router.face_detector_name = backend
        df_router.face_detector = \
            FaceDetector.build_model(df_router.face_detector_name)

    # Detect face regions (find_faces is an alias for detect_faces to avoid
    # possible name collisions)
    output = find_faces(image, detector_backend=df_router.face_detector_name,
                         align=align, return_type='regions',
                         face_detector=df_router.face_detector)
    regions = output['regions']
    
    # Draws rectangles over each detected face
    if len(regions) > 0:
        for roi in regions:
            image = cv2.rectangle(image,
                                  tuple(roi[0:2]),
                                  tuple(np.add(roi[0:2], roi[2:])),
                                  (0, 0, 255), 2)

    img = Image.fromarray(image[:, :, ::-1])
    im1 = img.save(myfile.filename)

    return FileResponse(myfile.filename)

# ------------------------------------------------------------------------------

@df_router.post("/detection/regions")
async def detect_regions(myfile: UploadFile, backend: FaceDetectorOptions,
                         align: bool = False):
    """
    API ENDPOINT: detect_regions
        Used to detect the region of each detected face. Works in images with 
        multiple faces. Returns a dictionary with a list o tuples where each
        tuple corresponds to a region: a 4 element tuple composed of:
            1. X coordinate of top left corner of the rectangular region
            2. Y coordinate of top left corner of the rectangular region
            3. X coordinate of bottom right corner of the rectangular region
            4. X coordinate of bottom right corner of the rectangular region
    """
    
    # Obtains contents of the file & transforms it into an image
    data  = np.fromfile(myfile.file, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Creates a face detector if one does not exist with the same name
    if df_router.face_detector_name != backend:
        df_router.face_detector_name = backend
        df_router.face_detector = \
            FaceDetector.build_model(df_router.face_detector_name)

    # Detects face regions (find_faces is an alias for detect_faces to avoid
    # possible name collisions)
    output = find_faces(image, detector_backend=df_router.face_detector_name,
                          align=align, return_type='regions',
                          face_detector=df_router.face_detector)
    regions = output['regions']
    
    # Construct output
    if len(regions) > 0:
        faces_output = Faces(faces=regions)
    else:
        faces_output = Faces(faces=[])

    return faces_output

# ------------------------------------------------------------------------------

@df_router.post("/detection/face")
async def detect_face(myfile: UploadFile, backend: FaceDetectorOptions,
                             align: bool = True):
    """
    API ENDPOINT: detect_face
        Used to detect a single face in a image. If there are multiple faces
        present, the first one to be detected is returned. Returns an image.
    """
    
    # Obtains contents of the file & transforms it into an image
    data  = np.fromfile(myfile.file, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Creates a face detector if one does not exist with the same name
    if df_router.face_detector_name != backend:
        df_router.face_detector_name = backend
        df_router.face_detector = \
            FaceDetector.build_model(df_router.face_detector_name)

    # Detect faces (find_faces is an alias for detect_faces to avoid possible
    # name collisions)
    output = find_faces(image, detector_backend=df_router.face_detector_name,
                         align=align, return_type='faces',
                         face_detector=df_router.face_detector)
    
    # Gets the first detected face and returns it
    face = output['faces'][0]
    img  = Image.fromarray(face[:, :, ::-1])
    im1  = img.save(myfile.filename)

    return FileResponse(myfile.filename)


# ______________________________________________________________________________
#                       VERIFICATION-RELATED API METHODS 
# ------------------------------------------------------------------------------

@df_router.post("/verification/verify")
async def face_verify(tgt_file: UploadFile, ref_file: UploadFile,
                      params: FaceVerifierParams):
    """
    API ENDPOINT: face_verify
        Used to verify if the person in image 1 (target) matches that of image 2
        (reference). Returns a JSON message with the 'verified' status (i.e.
        matches [True] or not [False]), distance between the representation of
        the two images, the threshold for acceptable similarity, the face
        verifier model, face detector model and the distance metric used to
        calculate the distance.

        Note: Please ensure that both images only contain a single person!
    """
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    tgt_data = np.fromfile(tgt_file.file, dtype=np.uint8)
    tgt_img  = cv2.imdecode(tgt_data, cv2.IMREAD_UNCHANGED)

    ref_data = np.fromfile(ref_file.file, dtype=np.uint8)
    ref_img  = cv2.imdecode(ref_data, cv2.IMREAD_UNCHANGED)

    # Builds the face verifier model
    df_router.face_verifier_name, metric_name, df_router.face_verifier = \
                build_face_verifier(model_name=params.model_name, 
                                    model=df_router.face_verifier,
                                    distance_metric=params.distance_metric)

    # Unnests single element list into element. Does the same for the dictionary
    # of models unless the model name is 'Ensemble' (as Ensemble models consists
    # of 4 models)
    df_router.face_verifier_name = df_router.face_verifier_name[0]
    metric_name                  = metric_name[0]

    if df_router.face_verifier_name != "Ensemble":
        df_router.face_verifier = \
            df_router.face_verifier[df_router.face_verifier_name]

    # Check if a face detection model has been used before, if not use a default
    # option
    if df_router.face_detector_name == None:
        use_backend = FaceDetectorOptions.OPENCV
    else:
        use_backend = df_router.face_detector_name

    # Verifies both faces
    response = verify_faces(tgt_img, img2_path=ref_img,
                    model_name = df_router.face_verifier_name,
                    distance_metric = metric_name,
                    model = df_router.face_verifier,
                    enforce_detection = params.enforce_detection,
                    detector_backend = use_backend,
                    align = params.align,
                    prog_bar = False,
                    normalization = params.normalization,
                    threshold = params.threshold)

    return response

# ------------------------------------------------------------------------------
@df_router.post("/verification/represent")
async def get_representation(myfile: UploadFile,
                             params: GetRepresentationParams):
    """
    API ENDPOINT: verify_faces
        Used to get the vector representation of a person in an image. Make sure
        the image provided contains only ONE person, otherwise, the
        representation of the first person detected will be returned. Returns a
        JSON message with ...
    """
    
    # Obtains contents of the target and reference files & transforms them into
    # images
    data = np.fromfile(myfile.file, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    # Builds the face verifier model
    df_router.face_verifier_name, _junk, df_router.face_verifier = \
                build_face_verifier(model_name=params.model_name, 
                                    model=df_router.face_verifier,
                                    distance_metric='cosine')

    # Unnests single element list into element. Does the same for the dictionary
    # of models unless the model name is 'Ensemble' (as Ensemble models consists
    # of 4 models)
    df_router.face_verifier_name = df_router.face_verifier_name[0]

    if df_router.face_verifier_name != "Ensemble":
        df_router.face_verifier = \
            df_router.face_verifier[df_router.face_verifier_name]

    # Check if a face detection model has been used before, if not use a default
    # option
    if df_router.face_detector_name == None:
        use_backend = FaceDetectorOptions.OPENCV
    else:
        use_backend = df_router.face_detector_name

    representation = DeepFace.represent(img, model_name = params.model_name,
                                model = df_router.face_verifier,
                                enforce_detection = params.enforce_detection,
                                detector_backend = use_backend,
                                align = params.align,
                                normalization = params.normalization)

    return {'representation':representation}

# ------------------------------------------------------------------------------

@df_router.post("/verification/distance")
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


# ______________________________________________________________________________
#                        RECOGNITION-RELATED API METHODS 
# ------------------------------------------------------------------------------

@df_router.post("/recognition/recognize/single") # uses deepface function 'find'
async def recognize_single_face(tgt_file: UploadFile,
                                params: FaceVerifierParams):
    """
    API ENDPOINT: recognize_single_face
        Recognizes a single image by comparing it to the images in the directory
        specified by the 'IMG_DIR' path variable.
    """    
    # Obtains contents of the target and reference files & transforms them into
    # images
    tgt_data = np.fromfile(tgt_file.file, dtype=np.uint8)
    tgt_img  = cv2.imdecode(tgt_data, cv2.IMREAD_UNCHANGED)

    # Builds the face verifier model
    df_router.face_verifier_name, metric_name, df_router.face_verifier = \
                build_face_verifier(model_name=params.model_name, 
                                    model=df_router.face_verifier,
                                    distance_metric=params.distance_metric)
    
    # Unnests single element list into element. Does the same for the dictionary
    # of models unless the model name is 'Ensemble' (as Ensemble models consists
    # of 4 models)
    df_router.face_verifier_name = df_router.face_verifier_name[0]
    metric_name                  = metric_name[0]

    if df_router.face_verifier_name != "Ensemble":
        df_router.face_verifier = \
            df_router.face_verifier[df_router.face_verifier_name]

    # Check if a face detection model has been used before, if not use a default
    # option
    if df_router.face_detector_name == None:
        use_backend = FaceDetectorOptions.OPENCV
    else:
        use_backend = df_router.face_detector_name

    # Runs face recognition
    response = DeepFace.find(tgt_img, img_dir, model_name=params.model_name,
                                distance_metric=params.distance_metric,
                                model=df_router.face_verifier,
                                enforce_detection=params.enforce_detection,
                                detector_backend=use_backend,
                                align=params.align, prog_bar=False,
                                normalization=params.normalization)

    #print(response)
    return response

# ------------------------------------------------------------------------------

