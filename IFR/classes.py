# ==============================================================================
#                                   API CLASSES
# ==============================================================================

from re                 import compile, IGNORECASE
from enum               import Enum
from uuid               import UUID
from typing             import List, Tuple, Optional
from pydantic           import BaseModel

from sqlalchemy import Table, Column, String, Integer, PickleType, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

# IMPLEMENTATION NOTE:
# Pydantic expects a dictionary by default. You can configure your model to also
# support loading from standard ORM parameters (i.e. attributes on the object
# instead of dictionary lookups) by setting:

# class Config:
#     orm_mode = True

# ______________________________________________________________________________
#                          ENUMERATION (OPTIONS) CLASSES
# ------------------------------------------------------------------------------

# Path parameter class for face detector name options
class FaceDetectorOptions(str, Enum):
    """
    Path parameter class: available face detector names
    """
    OPENCV = "opencv",
    SSD    = "ssd",
    MTCNN  = "mtcnn",
    RETINA = "retinaface"

# Path parameter class for face detector name options
class FaceVerifierOptions(str, Enum):
    """
    Path parameter class: available face verifier names
    """
    VGGFace    = "VGG-Face"
    FACENET    = "Facenet"
    FACENET512 = "Facenet512"
    OPENFACE   = "OpenFace"
    DEEPFACE   = "DeepFace"
    DEEPID     = "DeepID"
    ARCFACE    = "ArcFace"

# Path parameter class for distance metric options
class DistanceMetrics(str, Enum):
    """
    Path parameter class: available distance metrics.
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

# Path parameter class for image save types
class ImageSaveTypes(str, Enum):
    """
    Path parameter class: available image save types.
    """
    PNG = "png"
    JPG = "jpg"
    NPY = "npy"

# Path parameter class for message detail options
class MessageDetailOptions(str, Enum):
    """
    Path parameter class: available message detail options.
    """
    COMPLETE = "complete"  # all information
    SUMMARY  = "summary"   # summarized information

# Path parameter class for message output options
class MessageOutputOptions(str, Enum):
    """
    Path parameter class: available message output options.
    """
    STRUCTURE = "structure"  # as a JSON-friendly structure
    MESSAGE   = "message"    # as a "console printed" message

# Path parameter class for available representation properties
class AvailableRepProperties(str, Enum):
    """
    Path parameter class: available properties of the Representation class.
    """
    UNIQUE_ID  = "unique_id"
    IMAGE_NAME = "image_name"
    IMAGE_FP   = "image_fp"
    GROUP_NO   = "group_no"
    NAME_TAG   = "name_tag"
    REGION     = "region"
    EMBEDDINGS = "embeddings"

# ______________________________________________________________________________
#                                   DEFAULT VALUES
# ------------------------------------------------------------------------------

default_detector          = FaceDetectorOptions.RETINA
default_verifier          = FaceVerifierOptions.ARCFACE
default_metric            = DistanceMetrics.COSINE
default_normalization     = NormalizationTypes.BASE
default_image_save_type   = ImageSaveTypes.JPG
default_align             = True
default_enforce_detection = True
default_threshold         = -1
default_tags              = []
default_uids              = []
default_auto_grouping     = True
default_eps               = 0.5
default_min_samples       = 2

default_verbose           = False
default_msg_detail        = MessageDetailOptions.SUMMARY
default_msg_output        = MessageOutputOptions.STRUCTURE
default_property          = AvailableRepProperties.NAME_TAG

# ______________________________________________________________________________
#                       PATH PARAMETER (API INPUT) CLASSES
# ------------------------------------------------------------------------------

# Path parameter class for get representation options
class GetRepresentationParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the get representation options.
    """
    model_name       : FaceVerifierOptions = default_verifier
    enforce_detection: bool                = default_enforce_detection
    detector_backend : FaceDetectorOptions = default_detector
    align            : bool                = default_align
    normalization    : NormalizationTypes  = default_normalization

    class Config:
        orm_mode = True

# Path parameter class for face verification options
class FaceVerifierParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the face verification options.
    """
    model_name       : FaceVerifierOptions  = default_verifier
    distance_metric  : DistanceMetrics      = default_metric
    enforce_detection: bool                 = default_enforce_detection
    detector_backend : FaceDetectorOptions  = default_detector
    align            : bool                 = default_align
    normalization    : NormalizationTypes   = default_normalization
    threshold        : int                  = default_threshold

    class Config:
        orm_mode = True

# Path parameter class for create_database endpoint parameters
class CreateDatabaseParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the parameters required for database creation.
    """
    detector_name : FaceDetectorOptions       = default_detector
    verifier_names: List[FaceVerifierOptions] = [default_verifier]
    align         : bool                      = default_align
    normalization : NormalizationTypes        = default_normalization
    tags          : Optional[List[str]]       = default_tags
    uids          : Optional[List[str]]       = default_uids
    auto_grouping : Optional[bool]            = default_auto_grouping
    eps           : Optional[float]           = default_eps
    min_samples   : Optional[int]             = default_min_samples
    metric        : Optional[DistanceMetrics] = default_metric
    verbose       : bool                      = default_verbose

    class Config:
        orm_mode = True

# Path parameter class for 
class VerificationParams(BaseModel):
    """
    Path parameter class: defines the expected body request containing all of
    the parameters required for the verification / recognition process.
    """
    detector_name: FaceDetectorOptions = default_detector
    verifier_name: FaceVerifierOptions = default_verifier
    align        : bool                = default_align
    normalization: NormalizationTypes  = default_normalization
    metric       : DistanceMetrics     = default_metric
    threshold    : float               = default_threshold
    verbose      : bool                = default_verbose

    class Config:
        orm_mode = True

# ______________________________________________________________________________
#                          RESPONSE (API OUTPUT) CLASSES
# ------------------------------------------------------------------------------

# Response class for face regions
class Faces(BaseModel):
    """
    Response model class: defines the output for face regions.
    """
    faces: List[Tuple[int, int, int, int]]

# Response class for face verification matches
class VerificationMatch(BaseModel):
    """
    Response model class: defines the output for face verification match.
    """
    unique_id : UUID
    image_name: str
    group_no  : int
    name_tag  : str
    image_fp  : str
    region    : List[int]
    embeddings: List[str]
    distance  : float
    # threshold  : float

# Response class for representation summary output
class RepsSummaryOutput(BaseModel):
    """
    Response model class: defines the output for representation summary output.
    This class can be seen as a subset of RepsInfoOutput.
    """
    unique_id : UUID
    image_name: str
    group_no  : int
    name_tag  : str
    region    : List[int]

# Response class for representation information output
class RepsInfoOutput(BaseModel):
    """
    Response model class: defines the output for representation information
    output. This class can be seen as a superset of RepsSummaryOutput.
    """
    unique_id : UUID
    image_name: str
    group_no  : int
    name_tag  : str
    image_fp  : str
    region    : List[int]
    embeddings: List[str]

# ______________________________________________________________________________
#                                  CUSTOM CLASSES
# ------------------------------------------------------------------------------
# Used internally in functions, may have methods.

# Stores the (database) Representation of a face image
class Representation():
    """
    Class to store the model-specific embeddings for an image.
    
    Attributes:
        After __init__:
            1. unique_id  - unique identifier number that represents a face
            2. orig_name  - original image's name
            3. orig_fp    - original image's full path
            4. image_name - image name
            5. image_fp   - image full path
            6. group_no   - integer indicating which group / cluster this
                            Representation belongs to (-1 = no group / cluster)
            7. name_tag   - custom name given to the image
            8. region     - list of integers specifying the (rectangular) face
                            region on the original image (top-left corner,
                            bottom-right corner)
            9. embeddings - model-specific vector representation of the face
        
    Methods:
        1. show_info() - prints the object's information in a condensed,
            easy-to-read form
        2. show_summary() - prints a summary of the object's information in a
            one-liner
    """
    def __init__(self, unique_id, orig_name='', orig_fp='', image_name='',
                 image_fp='', group_no=-1, name_tag='', region=[],
                 embeddings={}):
        """
        Initializes the object with appropriate attributes
        """
        self.unique_id  = unique_id
        self.orig_name  = orig_name     # Currently not implemented in API
        self.orig_fp    = orig_fp       # Currently not implemented in API
        self.image_name = image_name
        self.image_fp   = image_fp
        self.group_no   = group_no
        self.name_tag   = name_tag
        self.region     = region
        self.embeddings = embeddings
        
    def show_info(self):
        """
        Shows detailed information about the Representation in a neat layout
        """
        print('Unique ID'.ljust(20) + f': {self.unique_id}',
              'Original name'.ljust(20) + f': {self.orig_name}',
              'Original full path'.ljust(20) + f': {self.orig_fp}',
              'Image name'.ljust(20) + f': {self.image_name}',
              'Group'.ljust(20) + f': {self.group_no}',
              'Tag'.ljust(20) + f': {self.name_tag}',
              'Image full path'.ljust(20) + f': {self.image_fp}',
              'Face region'.ljust(20) + f': {self.region}', sep='\n')
        
        if self.embeddings: # embeddings dictionary is NOT empty
            print('Embeddings:')
            for k, v in self.embeddings.items():
                print(f'  > {k}: [{v[0]}, {v[1]}, {v[2]}, ... , {v[-1]}]'\
                    + f'(len={len(v)})')
                
        else:
            print('No embedding found!')

    def show_summary(self):
        """
        Shows summarized information about the Representation in a one-liner
        """
        print(f'UID: {self.unique_id}'.ljust(25),
              f'Original name: {self.image_name}'.ljust(25),
              f'Image name: {self.image_name}'.ljust(25),
              f'Group: {self.group_no}'.ljust(15),
              f'Tag: {self.name_tag}'.ljust(15),
              f'Region: {self.region}'.ljust(15), sep=' | ')

# Database class for the Representations of several face images
class RepDatabase():
    """
    Class to create the database for storing the Representations of several
    face images.

    Note: in the documentation of this class, records are used to refer to
    Representations stored in this class.
    
    Attributes (after __init__):
            1. reps - list of Representations [list of Representations].
            2. size - tuple with the number of Representations (equal to
                len(reps)) [tuple].
        
    Methods:
         1. __init__()              - used to create the Database object.
         2. __str_is_uuid4__()      - used to determine if a string is a valid
                                        uuid4
         3. __str_is_image_name__() - used to determine if a string is a valid
                                        image name
         4. search()                - searches records in the database using
                                        UUIDs, string representations of valid
                                        UUIDs or image names.
         5. search_by_tag()         - searches records in the database using
                                        name tags.
         6. add_records()           - adds new record(s) to the database.
         7. remove_records()        - removes record(s) from the database.
         8. update_record()         - updates the information of 1 record of the
                                        database.
         9. clear()                 - clears the database, making it empty.
        10. rename_records_by_tag() - renames records by name tag.
        11. remove_from_group()     - removes records from a given group.
        12. edit_tag_by_group_no()  - edits the name tag of all records
                                        belonging to a group.
        13. sort_database()         - sorts the database based on a record's
                                        attribute (note that not all attributes
                                        are available for sorting).
        14. get_attribute()         - gets a specific attribute of all records
                                        in the database.
        15. view_database()         - prints information about each record in
                                        the database.
        16. view_by_group_no()      - gets all records that belong to a specific
                                        group.
    """
    def __init__(self, *reps: Representation):
        """
        Initializes the Representations Database (RepDatabase) class. Can be
        initialized as an empty database (if no Representation are provided).

        Attributes (after __init__):
            1. reps - list of Representations [list of Representations].
            2. size - tuple with the number of Representations (equal to
                len(reps)) [tuple].
        
        Input:
            1. reps - any number of Representation objects. For more information
                about them see help(Representation).
        
        Output:
            1. Updates the attributes.

        Signature:
            database = RepDatabase(rep1, rep2, rep3, ..., repN)
                                        OR
            database = RepDatabase()
        """
        self.reps = []
        for rep in reps:
            self.reps.append(rep)
        self.size = (len(self.reps))

    # -------------------------------------------------------------------------

    def __str_is_uuid4__(self, uuid_string):
        """
        Checks if the string provided 'uuid_string' is a valid uuid4 or not. The
        case is always ignored.

        Input:
            1. uuid_string - string representation of uuid including dashes
                ('-') [string].
    
        Output:
            1. boolean indicating if the string is a valid uuid4 or not.

        Signature:
            is_valid = string_is_valid_uuid4(uuid_string)
        """
        uuid_pattern = compile('^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?'
                                + '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z',
                                IGNORECASE)
        match        = uuid_pattern.match(uuid_string)
        return bool(match)

    # -------------------------------------------------------------------------

    def __str_is_image_name__(self, image_name_string, ignore_case=False):
        """
        Checks if the string provided 'image_name_string' is a valid image name
        or not. A valid image name is assumed to be a word followed by a
        '.'extension, e.g: my_image_001.jpg.

        Input:
            1. image_name_string - image name [string].

            2. ignore_case - toggles if the case should be ignored or not
                (default=False) [boolean].
    
        Output:
            1. boolean indicating if the string is a valid image name or not.

        Signature:
            is_valid = string_is_valid_image_name(image_name_string)
        """
        # Creates the appropriate regular expression pattern depending if the
        # case should be ignored
        if ignore_case:
            img_name_pattern = compile(r'[\w-]+\.[\w]+$', IGNORECASE)
        else:
            img_name_pattern = compile(r'[\w-]+\.[\w]+$')
    
        # Tries to find a match. If a match is found then the string is valid.
        # Otherwise it is not.
        match = img_name_pattern.match(image_name_string)

        return bool(match)

    # -------------------------------------------------------------------------

    def __records2resp_model__(self, records, amt_detail='complete'):
        """
        Converts a list of records (Representations) into a list of summarized
        or complete response models. See help(RepsSummaryOutput) and/or
        help(RepsInfoOutput) respectively for more information. The summarized
        or complete model is toggled through 'amt_detail'.

        If 'amt_detail' is neither 'summary' nor 'complete', then a Value Error
        is raised.

        Inputs:
            1. records - list of records [list of Representations].

            2. amt_detail - toggles between summarized or complete response
                models. Options: 'summarized' or 'complete' (default='complete')
                [string].
    
        Output:
            - If amt_detail='summary':
                List of RepsSummaryOutput response model objects.

            - If amt_detail='complete':
                List of RepsInfoOutput response model objects.

        Signature:
            output_objs = database.__records2resp_model__(records,
                                                          amt_detail='complete')
        """
        # Initializes output object
        output_obj = []

        # Case 1: SUMMARY
        if   amt_detail.lower() == 'summary':
            if len(records) > 0:
                for rec in records:
                    output_obj.append(RepsSummaryOutput(unique_id=rec.unique_id,
                                                    image_name=rec.image_name,
                                                    group_no=rec.group_no,
                                                    name_tag=rec.name_tag,
                                                    region=rec.region))

        # Case 2: COMPLETE
        elif amt_detail.lower() == 'complete':
            if len(records) > 0:
                for rec in records:
                    output_obj.append(RepsInfoOutput(unique_id=rec.unique_id,
                        image_name=rec.image_name, group_no=rec.group_no,
                        name_tag=rec.name_tag, image_fp=rec.image_fp,
                        region=rec.region,
                        embeddings=[name for name in rec.embeddings.keys()]))

        # Case 3: [Exception] UNKNOWN AMOUNT OF DETAIL
        else:
            raise ValueError('Amout of detail should be '\
                           + 'either SUMMARY or COMPLETE.')

        return output_obj

    # -------------------------------------------------------------------------

    def search(self, terms, get_index: bool = False):
        """
        Searches records in the database with terms in 'terms'. A term can be a
        single object or list of:
            1. UUID object(s) [UUID]
            2. string(s) representation(s) of a VALID UUID object(s) [string(s)]
            3. image name(s) [string]
        
        Note: a VALID UUID string representation obeys the following (case
        insensitive) regular expression:
                    '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
                    '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

        If 'terms' is a list, each element can be one of the 3 objects described
        above (i.e. one can mix and match objects in the 'terms' list).

        This functions returns the records (Representations) of each term found.
        If a term is invalid or does not match any record in the database, it is
        ignored / skipped.

        Inputs:
            1. terms - UUID, string representation of a valid UUID or image
                name. Either a single object or list containing any number of
                the aformentioned objects. You can mix and match. [UUID / string
                or list of UUIDs / strings].

            2. get_index - toggles if the function should return just the
                indexes of the Representations in the database (True) or if it 
                should return the Representations themselves (default=False)
                [boolean].

        Output:
            If get_index=False:
                List of records (Representations) that were matches to each term
                in 'terms' (in order). A invalid or inexistent term is skipped.

            If get_index=True:
                List of indexes (integers) that correspond to record
                (Representation) matches to each term in 'terms' (in order). A
                invalid or inexistent term is skipped.

        Signature:
            indexes      = database.search(terms, get_index=True)
                                        OR
            matched_reps = database.search(terms, get_index=False)
        """
        # Ensures that 'terms' is a list
        if not isinstance(terms, list):
            terms = [terms]

        # Obtains all valid terms from the 'terms' list provided
        val_terms = []
        for term in terms:
            # Term is a UUID (assumes the uid is valid)
            if   type(term) == UUID:
                val_terms.append(term)

            # Term is a valid string representation of a UUID
            elif self.__str_is_uuid4__(term):
                val_terms.append(UUID(term))

            # Term is a valid image name
            elif self.__str_is_image_name__(term):
                val_terms.append(term)

            else:
                pass # do nothing - skip invalid terms

        # Obtains either all indexes or entries (depending on 'get_index')
        output = []
        for i, rep in enumerate(self.reps):
            # If the unique id or the image name matches 1 of the valid terms,
            # append either the index or entry
            if (rep.unique_id  in val_terms) or (rep.image_name in val_terms):
                if get_index:
                    output.append(i)

                else:
                    output.append(rep)

            # Otherwise, skip it (i.e. do nothing)
            else:
                pass # do nothing

        return output

    # -------------------------------------------------------------------------

    def search_by_tag(self, tag: str, get_index: bool = False,
                      ignore_case: bool = False):
        """
        Searches the database based on a name tag provided. This functions
        returns a list with all records (Representations) in which their name
        tag match the 'tag' provided.

        Inputs:
            1. tag - name tag [string].

            2. get_index - toggles if the function should return just the
                indexes of the Representations in the database (True) or if it 
                should return the Representations themselves (default=False)
                [boolean].

            3. ignore_case - toggles if the function should ignore the case of
                the 'tag' during search (default=False) [boolean].

        Output:
            If get_index=False:
                List of all records (Representations) that have a name tag that
                matches the 'tag' provided.

            If get_index=True:
                List of all indexes (integers) that correspond to each record
                (Representation) that have a name tag that matches the 'tag'
                provided.

        Signature:
            indexes      = database.search_by_tag(terms, get_index=True,
                                                  ignore_case=False)
                                            OR
            matched_reps = database.search_by_tag(terms, get_index=False,
                                           ignore_case=False)
        """
        # Obtains either all indexes or entries (depending on 'get_index')
        output = []
        for i, rep in enumerate(self.reps):
            # Case 1: IGNORE CASE
            if ignore_case:
                # Case 1.1: IGNORE CASE & (MATCH FOUND) & GET INDEX
                if   rep.name_tag.lower() == tag.lower() and     get_index:
                    output.append(i)

                # Case 1.2: IGNORE CASE & (MATCH FOUND) & RETURN RECORD
                elif rep.name_tag.lower() == tag.lower() and not get_index:
                    output.append(rep)

                # Case 1.3: IGNORE CASE & (MATCH NOT FOUND)
                else:
                    pass # do nothing

            # Case 2: EXACT CASE
            else:
                # Case 2.1: EXACT CASE & (MATCH FOUND) & GET INDEX
                if   rep.name_tag == tag and     get_index:
                    output.append(i)

                # Case 2.2: EXACT CASE & (MATCH FOUND) & RETURN RECORD
                elif rep.name_tag == tag and not get_index:
                    output.append(rep)

                # Case 2.3: EXACT CASE & (MATCH NOT FOUND)
                else:
                    pass # do nothing

        return output

    # -------------------------------------------------------------------------

    def add_records(self, *reps: Representation):
        """
        Adds one or more records (Representations) to the database. Also updates
        the size of the database.

        Input:
            1. *reps - any number of Representation objects [Representation].

        Output:
            None

        Signature:
            database.add_records(rep1, rep2, ..., repN)
        """
        # Loops through each record, adding it to the database
        for rep in reps:
            self.reps.append(rep)

        # Updates the database's size (length)
        self.size = (len(self.reps))

    # -------------------------------------------------------------------------

    def remove_records(self, terms):
        """
        Removes one or more records (Representations) from the database. These
        records are searched by the term (or terms) in 'terms'. A term can a
        single object or list of:
            1. UUID object(s) [UUID]
            2. string(s) representation(s) of a VALID UUID object(s) [string]
            3. image name(s) [string]
        
        Note: a VALID UUID string representation obeys the following (case
        insensitive) regular expression:
                    '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
                    '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

        If 'terms' is a list, each element can be one of the 3 objects described
        above (i.e. one can mix and match objects in the 'terms' list). If a
        term is invalid or does not match any record in the database, it is
        ignored / skipped.
        
        Finally, this function also updates the size of the database.

        Input:
            1. terms - UUID, string representation of a valid UUID or image
                name. Either a single object or list containing any number of
                the aformentioned objects. You can mix and match. [UUID / string
                or list of UUIDs / strings].

        Output:
            1. Tuple containing the number of records removed and the number of
                terms skipped (skipped terms were invalid or did not have any
                matching records).

        Signature:
            (removed, skipped) = database.remove_records(terms)
        """
        # Determines the indexes of the terms that have a match. Terms with no
        # match are skipped.
        idxs = self.search(terms, get_index=True)

        # Loops through each index in reverse order (to avoid mismatched indexes
        # as we are removing them), removing the corresponding record.
        for idx in reversed(idxs):
            self.reps.pop(idx)

        # Updates the database's size (length)
        self.size = (len(self.reps))

        # Calculates the number of records removed and the number of records
        # skipped.
        removed = len(idxs)
        skipped = len(terms) - removed

        return (removed, skipped)

    # -------------------------------------------------------------------------

    def update_record(self, term, unique_id=None, image_name: str = None,
            image_fp: str = None, group_no: int = None, name_tag: str = None,
            region: List[tuple] = None, embeddings: dict = None):
        """
        Updates a single record (Representation) in the database. This record is
        searched by the term in 'term'. In this case, a term is a:
            1. UUID object [UUID]
            2. string representation of a VALID UUID object [string]
            3. image name [string]
        
        Note: a VALID UUID string representation obeys the following (case
        insensitive) regular expression:
                    '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
                    '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

        If the term provided is invalid (or does not have any matching record),
        then nothing will be updated.
        
        All optional parameters are attributes of a record (Representation) that
        can be modified. Only attributes that will be modified should be passed
        as arguments to this function.

        Important note: this function does not automatically determine the
        image's full path from the image's name as it does not know the image's
        directory. If the image name attribute is modified, make sure to modify
        the image full path attribute with the appropriate path as well, or this
        might lead to other errors!

        Input:
            1. term - UUID, string representation of a valid UUID or image
                name [UUID / string].

            2. unique_id - record's (Representation's) unique id [UUID].

            3. image_name - record's (Representation's) image name [string].
            
            4. image_fp - record's (Representation's) image full path [string].
            
            5. group_no - record's (Representation's) group number [integer].
            
            6. name_tag - record's (Representation's) name tag [string].
            
            7. region - record's (Representation's) region [list of 4 integers].
            
            8. embeddings - record's (Representation's) embeddings [dictionary].
        
        Output:
            1. Returns the updated record (Representation).

        Signature:
            updated_record = database.update_record(term, unique_id, image_name,
                                image_fp, group_no, name_tag, region,
                                embeddings)
        """
        # Finds the index for the specific record
        idx = self.search(term, get_index=True)
        idx = idx[0]

        # Updates the record's unique id if a new one was provided
        if unique_id  is not None:
            if   type(unique_id) == UUID:
                self.reps[idx].unique_id  = unique_id
            elif type(unique_id) == str:
                self.reps[idx].unique_id  = UUID(unique_id)
        
        # Updates the record's image name if a new one was provided
        if image_name is not None:
            self.reps[idx].image_name = image_name

        # Updates the record's image full path if a new one was provided
        if image_fp   is not None:
            self.reps[idx].image_fp   = image_fp

        # Updates the record's group number if a new one was provided
        if group_no   is not None:
            self.reps[idx].group_no   = group_no

        # Updates the record's name tag if a new one was provided
        if name_tag   is not None:
            self.reps[idx].name_tag   = name_tag

        # Updates the record's region if a new one was provided
        if region     is not None:
            self.reps[idx].region     = region

        # Updates the record's embeddings if a new one was provided
        if embeddings is not None:
            self.reps[idx].embeddings = embeddings

        return self.reps[idx]

    # -------------------------------------------------------------------------

    def clear(self):
        """
        Clears the database (making it empty). Also updates the database's size
        to zero.

        Input:
            None

        Output:
            None

        Signature:
            database.clear()
        """
        # Clears the database and sets its size to zero
        self.reps = []
        self.size = (0)

    # -------------------------------------------------------------------------

    def rename_records_by_tag(self, old_tag: str, new_tag: str,
                              ignore_case: bool = False):
        """
        Renames all records (Representations) that have a name tag which matches
        'old_tag', updating their name tag to 'new_tag'.

        Inputs:
            1. old_tag - name tag that records should have [string].

            2. new_tag - new name tag that records will be updated with
                [string].

            3. ignore_case - toggles if the function should ignore the case of
                the 'old_tag' during search (default=False) [boolean].

        Output:
            1. Returns all records (Representations) that had their name tag
                updated.
        
        Signature:
            updated_reps = database.rename_records_by_tag(old_tag, new_tag,
                                                          ignore_case)
        """
        # Finds the indexes of records that match the 'old_tag'
        idxs = self.search_by_tag(old_tag, get_index=True,
                                  ignore_case=ignore_case)

        # Loops through each record, editing its name tag
        output = []
        for idx in idxs:
            rep          = self.reps[idx]
            rep.name_tag = new_tag
            output.append(rep)

        return output

    # -------------------------------------------------------------------------

    def remove_from_group(self, terms):
        """
        Removes all records that match a term in 'terms', effectively setting
        their group number to -1 (i.e. no group / "groupless"). A term can be a
        single object or list of:
            1. UUID object(s) [UUID]
            2. string(s) representation(s) of a VALID UUID object(s) [string(s)]
            3. image name(s) [string]
        
        Note: a VALID UUID string representation obeys the following (case
        insensitive) regular expression:
                    '^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?' + 
                    '[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z'

        If 'terms' is a list, each element can be one of the 3 objects described
        above (i.e. one can mix and match objects in the 'terms' list). If a
        term is invalid or does not match any record in the database, it is
        ignored / skipped.

        Input:
            1. terms - UUID, string representation of a valid UUID or image
                name. Either a single object or list containing any number of
                the aformentioned objects. You can mix and match. [UUID / string
                or list of UUIDs / strings].

        Output:
            1. Tuple containing the number of records that had their group
                number removed and the number of terms skipped (skipped terms
                were invalid or did not have any matching records).

        Signature:
            (removed, skipped) = database.remove_from_group(terms)
        """
        # Determines the indexes of the terms that have a match. Terms with no
        # match are skipped.
        idxs = self.search(terms, get_index=True)

        # Loops through each index, setting each group number to -1 (i.e.
        # "groupless" / no group).
        for idx in idxs:
            self.reps[idx].group_no = -1

        # Calculates the number of records removed and the number of records
        # skipped.
        removed = len(idxs)
        skipped = len(terms) - removed

        return (removed, skipped)

    # -------------------------------------------------------------------------

    def edit_tag_by_group_no(self, target_group_no: int, new_name_tag: str):
        """
        Updates the name tag of all records (Representations) belonging to the
        specified group number 'target_group_no' with the new name tag
        'new_name_tag'.

        Inputs:
            1. target_group_no - target / desired group number [integer].

            2. new_name_tag - desired new name tag [string].

        Output:
            1. Returns the number of records (Representations) that had their
                name tags modified.

        Signature:
            modified = database.edit_tag_by_group_no(target_group_no,
                                                     new_name_tag)
        """
        # Initializes modified counter
        modified = 0

        # Gets the records in the database which correspond to the desired
        # group number
        if len(self.reps) == 0:   # no records
            return modified

        elif len(self.reps) >= 1: # one or many records
            for i in range(self.size):
                if target_group_no == self.reps[i].group_no:
                    self.reps[i].name_tag = new_name_tag
                    modified             += 1
                else:
                    pass # do nothing

        else: # this should never happen
            raise AssertionError('Representation database can '
                                +'not have a negative size!')

        return modified

    # -------------------------------------------------------------------------

    def sort_database(self, atr: str, reverse: bool = False):
        """
        Sorts the database based on the record's (Representation's) attribute
        'atr' specified. The database CANNOT be sorted by the 'region' and
        'embeddings' attributes. Attempting to sort them by these attributes, or
        providing an attribute that does not exist, will result in an Attribute
        Error.

        The database CAN be sorted using any of the following (record)
        attributes: unique_id, image_name, image_fp, group_no or name_tag.

        Note: the database is sorted in place.

        Inputs:
            1. atr - attribute name [string].

            2. reverse - sorts the database in descending order (True) or in
                ascending order (default=False) [boolean].

        Output:
            None

        Signature:
            database.sort_database(atr, reverse=False)

        """
        # Sorts the database according to the record's (Representation's)
        # attribute specified. If the attribute does not exist, is 'region' or
        # 'embeddings', then a Attribute Error is raised.
        if   atr.lower() == 'unique_id':
            self.reps.sort(key=lambda x: x.unique_id, reverse=reverse)
        elif atr.lower() == 'image_name':
            self.reps.sort(key=lambda x: x.image_name, reverse=reverse)
        elif atr.lower() == 'image_fp':
            self.reps.sort(key=lambda x: x.image_fp, reverse=reverse)
        elif atr.lower() == 'group_no':
            self.reps.sort(key=lambda x: x.group_no, reverse=reverse)
        elif atr.lower() == 'name_tag':
            self.reps.sort(key=lambda x: x.name_tag, reverse=reverse)
        elif atr.lower() in ['region', 'embeddings']:
            raise AttributeError("Can not sort database based on 'region' "
                               + "or 'embeddings' attributes")
        else:
            raise AttributeError()
        
    # -------------------------------------------------------------------------

    def get_attribute(self, atr: str, suppress_error: bool = True):
        """
        Gets the specified attribute 'atr' of all records (Representations) in
        the database. If 'suppress_error' is True, then instead of errors, this
        function returns empty lists. Otherwise, the function will raise an
        Attribute Error if the attribute requested does not exist.

        Inputs:
            1. atr - desired attribute [string].

            2. suppress_error - toggles if the function should return empty
                lists instead of errors (True) or not (default=True) [boolean].

        Output:
            1. List containing the desired output of each record
                (Representation) in the database. CHANGE THIS!

        Signature:
            attributes = database.get_attribute(atr, suppress_error=True)
        """
        # Initializes the output attributes object
        output_ats = []

        # Obtains all attributes available in a Representation (an entry)
        all_attributes = list(self.reps[0].__dict__.keys())

        if len(self.reps) == 0:
            # Returns an empty list if the database has no entries
            pass # do nothing
        
        elif len(self.reps) > 0 and atr in all_attributes:
            # Loops through all entries, obtaining the attribute if the
            # database has at least 1 entry and the attribute requested exists
            for rep in self.reps:
                if   atr == 'group_no':
                    output_ats.append(int(rep.__dict__[atr]))
                elif atr == 'region':
                    output_ats.append([int(item) for item in rep.__dict__[atr]])
                else:
                    output_ats.append(rep.__dict__[atr])

        elif suppress_error:
            # If suppress error is True and:
            #   a. database has a negative length (should never happen)
            #   b. attribute requested does not exist / is not valid
            # Returns an empty list
            pass # do nothing
        
        else:
            # Otherwise, checks if the database has a negative length (should
            # never happen) and if that is False, then raises a attribute error
            # as the attribute requested does not exist (and
            # suppress_error=False)
            assert len(self.reps) > 0
            raise AttributeError('Attribute does not exist!')

        return output_ats
    
    # -------------------------------------------------------------------------

    def view_database(self, detail: str = 'complete'):
        """
        Prints information about each record (Representation) in the database.
        This is useful to "visualize" the database. If detail='complete' then
        all the information of each record is printed. If detail='summary' then
        only a summarized information of each record is printed. Otherwise, an
        Assertion Error is raised (as detail has to be either 'complete' or
        'summary').

        Input:
            1. detail - either 'complete' (for all information) or 'summary'
                (for summarized information) (default='complete') [string].

        Output:
            None

        Signature:
            database.view_database(detail='complete')
        """
        # Case 1: SUMMARY
        if   detail == 'summary':
            if len(self.reps) > 0:
                for rep in self.reps:
                    rep.show_summary()
                    print('\n', '-'*75, '\n')

        # Case 2: STRUCTURE & COMPLETE
        elif detail == 'complete':
            if len(self.reps) > 0:
                for rep in self.reps:
                    rep.show_info()
                    print('\n', '-'*75, '\n')

        # Case 3: [Exception] STRUCTURE & ???
        else:
            raise AssertionError("Amount of detail should be "\
                               + "either 'summary' or 'complete'.")

    # -------------------------------------------------------------------------

    def view_by_group_no(self, target_group_no: int, print_info: bool = False):
        """
        Returns all records (Representations) belonging to the desired group
        'target_group_no'. Also prints all the information related to each
        record found if print_info=True.

        Input:
            1. target_group_no - target / desired group number [integer].

            2. print_info - toggles if all the information relating to each
                found record (Representation) should also be printed (True) or
                not (default=False) [boolean].

        Output:
            1. List of all records (Representations) belonging to the specified
                group number.

        Signature:
            reps_found = database.view_by_group_no(target_group_no,
                                                   print_info=False)
        """
        # Initialize output object
        reps_found = []

        # Gets the records in the database which correspond to the desired
        # group number
        if len(self.reps) == 0:   # no records
            return reps_found

        elif len(self.reps) >= 1: # one or many records
            for rep in self.reps:
                if target_group_no == rep.group_no:
                    reps_found.append(rep)

                    # Prints the record's information if 'print_info' is True
                    if print_info:
                        rep.show_info()
                    else:
                        pass # do nothing
                else:
                    pass # do nothing

        else: # this should never happen
            raise AssertionError('Representation database can '
                                +'not have a negative size!')

        return reps_found

# ______________________________________________________________________________
#                         SQLALCHEMY TABLES DEFINITIONS
# ------------------------------------------------------------------------------

Base = declarative_base()

class Person(Base):
    """
    Person data structure

    Fields:
        person_id
        name
        note
    """
    # Table name
    __tablename__ = 'person'

    # Object attributes (as database columns)
    id   = Column(Integer, primary_key=True)
    name = Column(String)
    note = Column(String)

    # Establishes connection to associated Face Representations
    reps = relationship("FaceRep", back_populates="person",
                        cascade="all, delete, delete-orphan") # important for deleting children
    
    # Standard repr for the class
    def __repr__(self):
        return "(id=%s) - %s\n%s" % (self.id, self.name, self.note)

class FaceRep(Base):
    """
    Initializes the object with appropriate attributes
    """
    # Table name
    __tablename__ = 'representation'

    # Object attributes (as database columns)
    id              = Column(Integer, primary_key=True)
    person_id       = Column(Integer, ForeignKey('person.id'))
    image_name_orig = Column(String(100))
    image_name      = Column(String(100))
    image_fp_orig   = Column(String(255))
    image_fp        = Column(String(255))
    group_no        = Column(Integer)
    region          = Column(PickleType)
    embeddings      = Column(PickleType)

    # Establishes connection to associated Person
    person = relationship("Person", back_populates="reps")

    # Standard repr for the class
    def __repr__(self):
        return "(id=%s)\nimage name: %s\nimage path: %s\ngroup: %s" % (self.id,
                    self.image_name, self.image_fp, self.group_no)
