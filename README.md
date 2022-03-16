## Loki (face recognition project)

The objective of this project is to implement a face recognition (FR) solution like the one present in Google photos.


# Progress checklist
 - [x] Phase 1: Face Detection
 - [x] Phase 2: Face Alignment
 - [x] Phase 3: Face Recognition
 - [x] Creation of proof of concepts (POCs) demonstrations
 - [ ] Creation of Restful API using FastAPI


# Face detector app

The face detector app is located inside the `./face_detector_app` directory. If you prefer to use Conda to manage your
environments, create a new environment using the `environment.yml`:

    conda env create -f environment.yml

Alternatively, if you prefer to manage your virtual environments using virtualenv, you can also install using pip and
the `requirements.txt`:

    pip install -r requirements.txt

The face detector is fully contained in a single file. Two images are provided to test the functionality of the app.


# Face recognition api

The face recognition api is located inside the `./api` directory. If you prefer to use Conda to manage your
environments, create a new environment using the `environment.yml`:

    conda env create -f environment.yml

Alternatively, if you prefer to manage your virtual environments using virtualenv, you can also install using pip and
the `requirements.txt`:

    pip install -r requirements.txt

To run the face recognition api we suggest reading through fastapi's [Run a Server Manually - Uvicorn documentation page](https://fastapi.tiangolo.com/deployment/manually/). Uvicorn was used during the development of this api and as such is the recommended alternative.
It can be installed, if needed, with the usual `pip` command:

    pip install "uvicorn[standard]"

Once uvicorn is installed, the api can be accesed by changing the current working directory to the root directory containing the api project and running the following command:

    uvicorn api.app:app

The api has the following structure:
```
api
 ├── FR_developing.ipynb
 ├── __init__.py
 ├── api_classes.py
 ├── api_functions.py
 ├── app.py
 ├── data
 │   ├── database
 │   ├── gallery
 │   ├── raw
 │   └── targets
 ├── environment.yml
 ├── face_recognition_pipeline.ipynb
 ├── global_variables.py
 ├── requirements.txt
 ├── routers
 │   ├── __init__.py
 │   ├── attribute_analysis.py
 │   ├── detection.py
 │   ├── recognition.py
 │   └── verification.py
 └── saved_models
     └── verifiers
```

Most of the subdirectories and files use is contained in their names: `./routers` subdirectory contains different API routers responsible for different overarching tasks, like face detection, verification, recognition and attribute analysis.

The `./saved_models` subdirectory stores all pre-built models. For now, verifier and attribute analysis models are stored in the `./save_models/verifiers` subdirectory (need to seperate attribute analysis models from this). The built models get loaded onto memory when the api starts to speed up the subsequent api calls (instead of building each model everytime a specific endpoint is accessed).

All classes, functions and global variables used are conveniently stored in the `api_classes.py`, `api_functions.py` and `global_variables.py` files respectively.

The main application is contained in the `app.py` file.


# Demos
 - **[face_detection_with_retinaface](demos/face_detection_with_retinaface.ipynb)** - Demonstrates face detection using RetinaFace and downloaded online image. This demo is fully explained and commented. GoogleColab version - no setup required, hooray!

 - **[GC_functionality_demo](demos/GC_deepface_functionality.ipynb)** - illustrates several different functionalities
 using the deepface framework, including face detection, alignment, verification/recognition, attribute analysis.
 GoogleColab version - no setup required, hooray!

 - **[GC_lfw_face_recognition](demos/GC_lfw_face_recognition.ipynb)** - evaluates the performance of the face
 recognition system on a subset of the Labelled Faces in the Wild (LFW) data set. The confusion matrix is provided along
 with other important metrics such as accuracy, false positive, false negative, etc. GoogleColab version - no setup
 required, hooray!

 - **[GC_use_your_dataset_FR](demos/GC_use_your_dataset_FR.ipynb)** - allows the user to upload images to a *gallery*
 (reference) and *target* directories. The face recognition system is then executed and the matches between the *target*
 images and the *gallery* are shown (if available). GoogleColab version - no setup required, hooray!

 - **[detailed_FR_pipeline](demos/detailed_FR_pipeline.ipynb)** - shows the  step-by-step process (pipeline) for the
 face recognition system.

 - **[FR_complete_demo](demos/GC_FR_complete_demo.ipynb)** - complete face recognition demo:
    - Automatically downloads a sample dataset from *deepface* repo
    - Splits it into *database* and *target* sets
    - Creates a database from the - you guessed it - *database* set
    - Runs face verification of each image in the *target* set agasint the database
    - Adds the verified images to the database
    - Downloads, recognizes and exposes the results of images from the web
    - Compares the performance (time taken) when using the Faiss library and when using a vectorized distance
    calculation.
    - GoogleColab version - no setup required, hooray!

