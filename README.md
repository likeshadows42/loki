## Loki (face recognition project)

The objective of this project is to implement a face recognition (FR) solution 
like the one present in Google photos.

# Progress checklist
 - [x] Phase 1: Face Detection
 - [x] Phase 2: Face Alignment
 - [x] Phase 3: Face Recognition

# Face detector app

The face detector app is located inside the *./face_detector_app* directory. If you prefer to use Conda to manage your
environments, create a new environment using the *environment.yml*:

    conda env create -f environment.yml

Alternatively, if you prefer to manage your virtual environments using virtualenv, you can also install using pip and
the *requirements.txt*:

    pip install -r requirements.txt

The face detector is fully contained in a single file. Two images are provided to test the functionality of the app.

# Demos
 - **[face_detection_with_retinaface](demos/face_detection_with_retinaface.ipynb)** - self-contained Google Colab
 demonstrating face detection using RetinaFace and image downloaded online. This demo is fully explained and commented.

 - **[GC_functionality_demo](demos/GC_deepface_functionality.ipynb)** - illustrates several different functionalities
 using the deepface framework, including face detection, alignment, verification/recognition, attribute analysis.
 GoogleColab version - no setup required, hooray!

 - **[GC_lfw_face_recognition](demos/GC_lfw_face_recognition.ipynb)** - evaluates the performance of the face
 recognition system on a subset of the Labelled Faces in the Wild (LFW) data set. The confusion matrix is provided along
 with other important metrics such as accuracy, false positive, false negative, etc. GoogleColab version - no setup
 required, hooray!

  - **[GC_use_your_dataset_FR](demos/GC_use_your_dataset_FR.ipynb)** - allows the user to upload images to a 'gallery'
  (reference) and target directories. The face recognition system is then executed and the matches between the target
  images and the gallery are shown (if available). GoogleColab version - no setup required, hooray!

  - **[detailed_FR_pipeline](demos/detailed_FR_pipeline.ipynb)** - shows the  step-by-step process (pipeline) for the
  face recognition system.
