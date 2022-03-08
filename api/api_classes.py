# ==============================================================================
#                                   API CLASSES
# ==============================================================================

from enum               import Enum
from typing             import List, Tuple
from pydantic           import BaseModel

# ------------------------------------------------------------------------------
#                 FACE RECOGNITION & ATTRIBUTE ANALYSIS CLASSES
# ------------------------------------------------------------------------------

# Path parameter class for face detector name options
class FaceDetectorOptions(str, Enum):
    OPENCV = "opencv",
    SSD    = "ssd",
    DLIB   = "dlib",
    MTCNN  = "mtcnn",
    RETINA = "retinaface"

# Response class for face regions
class Faces(BaseModel):
    faces: List[Tuple[int, int, int, int]]

# Path parameter class for face detector name options
class FaceVerifierOptions(str, Enum):
    """
    Path parameter class: available face detector names
    """
    VGGFace    = "VGG-Face"
    FACENET    = "Facenet"
    FACENET512 = "Facenet512"
    OPENFACE   = "OpenFace"
    DEEPFACE   = "DeepFace"
    DEEPID     = "DeepID"
    ARCFACE    = "ArcFace"
    DLIB       = "Dlib"

# Path parameter class for distance metric options
class DistanceMetrics(str, Enum):
    """
    Path parameter class: available distance metrics
    """
    COSINE       = "cosine"
    EUCLIDEAN    = "euclidean"
    EUCLIDEAN_L2 = "euclidean_l2"

# Path parameter class for image normalization options
class NormalizationTypes(str, Enum):
    """
    Path parameter class: available image normalizations. Certain normalizations
    improve the face verification accuracy depending on the verifier model used.
    Usually the normalization has a similar or matching name to the verifier
    model (e.g. FACENET normalization improves the FACENET model).
    """
    BASE        = "base"
    RAW         = "raw"
    FACENET     = "Facenet"
    FACENET2018 = "Facenet2018"
    VGGFACE     = "VGGFace"
    VGGFACE2    = "VGGFace2"
    ARCFACE     = "ArcFace"

# Path parameter class for get representation options
class GetRepresentationParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the get representation options.
    """
    model_name: FaceVerifierOptions = FaceVerifierOptions.VGGFace
    enforce_detection: bool = True
    detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV
    align: bool = True
    normalization: NormalizationTypes = NormalizationTypes.BASE

    # Pydantic expects a dictionary by default. You can configure your model to
    # also support loading from standard ORM parameters (i.e. attributes on the
    # object instead of dictionary lookups):
    class Config:
        orm_mode = True

# Path parameter class for face verification options
class FaceVerifierParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the face verification options.
    """
    model_name: FaceVerifierOptions = FaceVerifierOptions.VGGFace
    distance_metric: DistanceMetrics = DistanceMetrics.COSINE
    enforce_detection: bool = True
    detector_backend: FaceDetectorOptions = FaceDetectorOptions.OPENCV
    align: bool = True
    normalization: NormalizationTypes = NormalizationTypes.BASE
    threshold: int = -1

    # Pydantic expects a dictionary by default. You can configure your model to
    # also support loading from standard ORM parameters (i.e. attributes on the
    # object instead of dictionary lookups):
    class Config:
        orm_mode = True

# Path parameter class for image save types
class ImageSaveTypes(str, Enum):
    """
    Path parameter class: available image save types.
    """
    PNG = "png"
    JPG = "jpg"
    NPY = "npy"

# Stores the representation (embeddings) of a face image
class Representation():
    """
    Class to store the model-specific embeddings for an image.
    
    Attributes:
        After __init__:
            1. unique_id  - unique identifier number that represents a face
            2. image_name - image name
            3. image_fp   - image full path
            4. embeddings - model-specific vector representation of the face
        
    Methods:
        1. show_info() - prints the representation information in a condensed,
            easy-to-read form
    """
    def __init__(self, unique_id, image_name='', image_fp='', name_tag='',
                 region=[], embeddings={}):
        self.unique_id  = unique_id
        self.name_tag   = name_tag
        self.image_name = image_name
        self.image_fp   = image_fp
        self.region     = region
        self.embeddings = embeddings
        
    def show_info(self):
        print('Unique ID'.ljust(25) + f': {self.unique_id}',
              'Name'.ljust(25) + f': {self.name_tag}',
              'Image name'.ljust(25) + f': {self.image_name}',
              'Image full path'.ljust(25) + f': {self.image_fp}',
              'Face region'.ljust(25) + f': {self.region}', sep='\n')
        
        if self.embeddings: # embeddings dictionary is NOT empty
            print('Embeddings:')
            for k, v in self.embeddings.items():
                print(f'  > {k}: [{v[0]}, {v[1]}, {v[2]}, ... , {v[-1]}]'\
                    + f'(len={len(v)})')
                
        else:
            print('No embedding found!')

#
class FRMatch():
    def __init__(self):
        self.verified = False
        self.matches  = []

    def add_new_match(self, match_obj):
        if match_obj['verified'] == True:
            self.matches.append(match_obj)
            self.matches.sort(key=lambda k: k['distance'])
            self.verified = True