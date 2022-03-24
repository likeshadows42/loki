# ==============================================================================
#                         DETECTION-RELATED API METHODS
# ==============================================================================
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy            as np

from PIL                import Image
from fastapi            import APIRouter, UploadFile
from fastapi.responses  import FileResponse
from deepface.detectors import FaceDetector
from IFR.functions      import detect_faces as find_faces
from IFR.classes        import FaceDetectorOptions, Faces

# --------------------------- ROUTER INITIALIZATION ----------------------------
fd_router                    = APIRouter()
fd_router.face_detector      = None
fd_router.face_detector_name = None

# -------------------------------- API METHODS ---------------------------------
@fd_router.post("/detect/overlay_faces")
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
    if fd_router.face_detector_name != backend:
        fd_router.face_detector_name = backend
        fd_router.face_detector = \
            FaceDetector.build_model(fd_router.face_detector_name)

    # Detect face regions (find_faces is an alias for detect_faces to avoid
    # possible name collisions)
    output = find_faces(image, detector_backend=fd_router.face_detector_name,
                         align=align, return_type='regions',
                         face_detector=fd_router.face_detector)
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
@fd_router.post("/detect/regions")
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
    if fd_router.face_detector_name != backend:
        fd_router.face_detector_name = backend
        fd_router.face_detector = \
            FaceDetector.build_model(fd_router.face_detector_name)

    # Detects face regions (find_faces is an alias for detect_faces to avoid
    # possible name collisions)
    output = find_faces(image, detector_backend=fd_router.face_detector_name,
                          align=align, return_type='regions',
                          face_detector=fd_router.face_detector)
    regions = output['regions']
    
    # Construct output
    if len(regions) > 0:
        faces_output = Faces(faces=regions)
    else:
        faces_output = Faces(faces=[])

    return faces_output

# ------------------------------------------------------------------------------
@fd_router.post("/detect/face")
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
    if fd_router.face_detector_name != backend:
        fd_router.face_detector_name = backend
        fd_router.face_detector = \
            FaceDetector.build_model(fd_router.face_detector_name)

    # Detect faces (find_faces is an alias for detect_faces to avoid possible
    # name collisions)
    output = find_faces(image, detector_backend=fd_router.face_detector_name,
                         align=align, return_type='faces',
                         face_detector=fd_router.face_detector)
    
    # Gets the first detected face and returns it
    face = output['faces'][0]
    img  = Image.fromarray(face[:, :, ::-1])
    im1  = img.save(myfile.filename)

    return FileResponse(myfile.filename)

# ------------------------------------------------------------------------------
