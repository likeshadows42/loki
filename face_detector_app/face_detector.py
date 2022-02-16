# Module / package imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from tkinter import * 
from tkinter import filedialog as fd 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

import os
import numpy              as np
import matplotlib.pyplot  as plt
import matplotlib.image   as mpimg

from deepface.commons        import functions
from deepface.detectors      import FaceDetector
from matplotlib.widgets      import RectangleSelector, Slider

# ==============================================================================
#                             FUNCTIONS & CLASSES
# ==============================================================================

def detect_faces(img_path, detector_backend = 'opencv',
                 align = True, return_type = 'both'):
    """
    Detects faces in an image (and optionally aligns them).
    
    Inputs:
        1. img_path - image path, base64 image or numpy array image
        2. detector_backend - string corresponding to detector ([opencv],
            ssd, dlib, mtcnn, retinaface, mediapipe).
        3. align - flag indicating if face should be aligned ([align=True]).
        4. return_type - string indicating if faces, regions or both should
            be returned ('faces', 'regions', ['both']).
            
    Outputs:
        If return_type='regions':
            Dictionary containing list of face detections. The face detections
            (or regions of interest - rois) are lists with the format
            [top-left x, top-left y, width, height]. The dictionary key is
            'regions'.
            
        If return_type='faces':
            Dictionary containing list of detected faces. Each detection is an
            image with 'target_size' size (the number of color channels is
            unchanged). The dictionary key is 'faces'.
            
        If return_type='both':
            Dictionary containing both of the above outputs: face detections and
            detected faces. The dictionary keys are 'faces' and 'regions'. 
    
    Signature:
        output = detect_faces(img_path, detector_backend = 'opencv',
                              align = True, return_type = 'both')
    """
    # Raises an error if return type is not 'faces', 'regions' or 'both'.
    # Otherwise, initializes lists.
    if return_type == 'faces' or return_type == 'regions' or \
        return_type == 'both':
        faces = []
        rois  = []
    else:
        raise ValueError("Return type should be 'faces', 'regions' or 'both'.")
    
    # Loads image. Image might be path, base64 or numpy array. Convert it to
    # numpy whatever it is.
    img = functions.load_image(img_path)

    # The detector is stored in a global variable in FaceDetector object.
    # This call should be completed very fast because it will return found in
    # memory and it will not build face detector model in each call (consider
    # for loops)
    face_detector = FaceDetector.build_model(detector_backend)
    detections    = FaceDetector.detect_faces(face_detector, detector_backend,
                                              img, align)

    # Prints a warning and returns an empty dictionary an error if no faces were
    # found, otherwise processes faces & regions
    if len(detections) == 0:
        print('Face could not be detected or the image contains no faces.')

    else:
        # Loops through each face & region pair
        for face, roi in detections:

            # Only process images (faces) if the return type is 'faces'
            # or 'both'
            if return_type == 'faces' or return_type == 'both':
                # Appends processed face
                faces.append(face)
    
            # Only process regions (rois) if the return type is 'regions'
            # or 'both'
            if return_type == 'regions' or return_type == 'both':
                rois.append(roi)

  # ------------------------
  
    if return_type == 'faces':
        return {'faces':faces}
    elif return_type == 'regions':
        return {'regions':rois}
    else:
        assert return_type == 'both', "Return type should be 'both' here."
        return {'faces':faces, 'regions':rois}

# ==============================================================================

def get_cur_coords_callback(eclick, erelease):
    """
    Callback to get the cursor's start and end coordinates on button press and
    release events. Stores the results in 'toggle_selector.cur_coords' as a
    [x1, y1, x2, y2] list

    Input:
        1. eclick   - mouse click press (press) event
        2. erelease - mouse click release (release) event

    Output:
        1. toggle_selector.cur_coords

    Signature:
        get_cur_coords_callback(eclick, erelease)
    """

    x1, y1 = eclick.xdata, eclick.ydata     # first coordinates, mouse press
    x2, y2 = erelease.xdata, erelease.ydata # second coordinates, mouse release
    toggle_selector.cur_coords = [x1, y1, x2, y2]

# ==============================================================================

def toggle_selector(event):
    """
    This function will act as an object in the sense that it will contain
    attributes. These attributes are used to pass the required variables to this
    function's local scope without requiring them as inputs.

    Controls the key press interactivity:
      1. 'F' or 'f': deactivates the manual selector and updates any
                     selections made

      2. 'enter'   : generates a region of interest from the manual selection,
                     draws it and stores it

    Input:
        1. event
        
    Uses:
        1. toggle_selector object

    Output:
        1. toggle_selector object

    Signature:
        toggle_selector(event)
    """

    if event.key in ['F', 'f'] and toggle_selector.RS.active:
        print('Selector deactivated.')
        toggle_selector.RS.set_active(False)

        if len(toggle_selector.coords) > 0:
            toggle_selector.update_func(toggle_selector.coords)
        
        toggle_selector.RS = None
        toggle_selector.button.config(state=NORMAL)

    elif event.key.lower() == 'enter':
        roi = np.array([max(np.floor(toggle_selector.cur_coords[0]), 0),
                        max(np.floor(toggle_selector.cur_coords[1]), 0),
                        np.ceil(toggle_selector.cur_coords[2]),
                        np.ceil(toggle_selector.cur_coords[3])], dtype=int)

        roi[2:] = roi[2:] - roi[:2]

        if toggle_selector.cur_coords != None:
            if len(toggle_selector.coords) == 0 \
                or not np.any(np.all(roi == toggle_selector.coords, axis=1)):
                toggle_selector.coords.append(roi)

                # Create a Rectangle patch
                rect = Rectangle((roi[0], roi[1]), roi[2], roi[3],
                                 linewidth=1, edgecolor='r',
                                 facecolor='none')
            
                # Add the patch to the Axes
                toggle_selector.ax_handle.add_patch(rect)
                toggle_selector.chart_type.draw()

    # print(f'[toggle_selector] Coords: {toggle_selector.coords}')

# ==============================================================================

def update_slider(val):
    """
    This function (callback) will act as an object in the sense that it will
    contain attributes. These attributes are used to pass the required variables
    to this function's local scope without requiring them as inputs.

    Updates the graph / plot based on the slider's state (value).

    Input:
        1. val - numerical value (is convert to integer i.e., 'discretized')
        
    Uses:
        1. update_slider.ax - axes handle (this axes will be updated)
        2. update_slider.chart_type - chart type handle (used to force axes to
                graphically update)

    Output:
        None (updates axes)

    Signature:
        update_slider(val)
    """
    val = int(val) # discretizes value by casting it into integer
    update_slider.ax.imshow(update_slider.data[val][:, :, ::-1]) # updates graph
    update_slider.ax.set_title(f'Detected face {val+1}')         # updates title
    update_slider.chart_type.draw() # forces graphical update

# ==============================================================================


class Face_Detector_App:

    def __init__(self, master):
        """
        Sets up (builds) the entire face detector app.

        Input:
            1. master - Tk ('root') object (base object)

        Output:
            None (builds and displays the app)

        Signature:
            Face_Detector_App(root)
        """
        # Sets up the master, app's start path and frames (panels)
        self.master       = master
        self.start_path   = os.path.dirname(os.path.realpath("__file__"))
        self.image_panel  = Frame(self.master, bg='#5BA8FF', width=500,
                                  height=500, relief='ridge', borderwidth=2) # left area
        self.result_panel = Frame(self.master, bg='#C7C7C7', width=300,
                                  height=500, relief='ridge', borderwidth=2) # right area

        # Layout all of the main containers
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        self.image_panel.grid(row=0, column=0, sticky="nw")
        self.result_panel.grid(row=0, column=1, sticky="ne")

        # Figure and axes (result panel) [fig2]
        self.fig2 = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax2  = self.fig2.add_subplot(111)
        self.chart_type2 = FigureCanvasTkAgg(self.fig2, self.result_panel)
        self.chart_type2.get_tk_widget().grid(row=0, column=0, columnspan=3)
        self.ax2.set_title('')

        # Slider place holder
        self.ax_slider = self.fig2.add_axes([0.20, 0.01, 0.65, 0.03])

        # Figure and axes (image panel) [fig]
        self.fig = plt.Figure(figsize=(6, 4.5), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.chart_type = FigureCanvasTkAgg(self.fig, self.image_panel)
        self.chart_type.get_tk_widget().grid(row=0, column=0, columnspan=5)
        self.ax.set_title('No image loaded')

        # Open image button
        self.open_image_button = Button(self.image_panel, text = 'Open Image',
                                      bd = 2, fg = 'white', bg = '#5BA8FF',
                                      font = ('', 15), command=self.open_image)
        self.open_image_button.grid(row=1, column=0, rowspan=2)

        # Face detector label
        self.face_detector_label = Label(self.image_panel, fg='white',
                                         bg='#5BA8FF', text='Face detector:',
                                         font = ('', 12))
        self.face_detector_label.grid(row=1, column=1)
    
        # Choose face detector drop down
        self.face_detector_opts = ['opencv', 'ssd', 'dlib',
                                    'mtcnn', 'retinaface']
        self.face_detector = StringVar()
        self.face_detector.set(self.face_detector_opts[0])
        self.fd_dd = OptionMenu(self.image_panel, self.face_detector,
                                *self.face_detector_opts)
        self.fd_dd.config(bg='#5BA8FF')
        self.fd_dd.grid(row=2, column=1)

        # Detect faces button
        self.detect_faces_button = Button(self.image_panel,
                    text = 'Detect faces', bd = 2, fg = 'white', bg = '#5BA8FF',
                    font = ('', 12), command=self.detect_faces, state=DISABLED)
        self.detect_faces_button.grid(row=1, column=3, rowspan=2)

        # Enable manual selector button
        self.enable_manual_selector_button = Button(self.image_panel,
                    text = 'Enable manual selector', bd = 2, fg = 'white',
                    bg = '#5BA8FF', font = ('', 12),
                    command=self.enable_manual_selector, state=DISABLED)
        self.enable_manual_selector_button.grid(row=1, column=4, rowspan=2)

        # Initializing attributes
        self.face_detector_results = None
        self.manual_coords         = []
        self.loaded_img            = None

        # Print to separate GUI console output from any initialization warnings
        print('')

    # --------------------------------------------------------------------------

    def open_image(self):
        """
        Open image button. Opens a PNG or JPG image, plots it and adds its name
        as title.

        Inputs:
            1. self
        
        Uses attributes:
            1. self.ax
            2. self.chart_type

        Output:
            1. self.filename
        """
        # This try statement catches the exception if the user cancels the file
        # selection process. On exception, ignore it (pass)
        try:
            # Asks the user to choose a file
            self.filename = fd.askopenfile(initialdir=self.start_path, 
                                title='Select a image file',
                                filetypes=(('All files', ['*.png', '*.jpg']),
                                ('PNG', '*.png'), ('JPG', '*.jpg')))

            # Resets all axes
            self.ax.clear()
            self.ax2.clear()
            self.ax_slider.clear()
            self.chart_type.draw()
            self.chart_type2.draw()
            
            # Stores and plots the chosen image
            self.loaded_img = mpimg.imread(self.filename.name)
            self.ax.imshow(self.loaded_img)
            self.ax.set_title(self.filename.name.split("/")[-1])
            self.chart_type.draw()

            # Once there is an image to process, enable the manual selector and
            # the face detetor button
            self.detect_faces_button.config(state=NORMAL)
            self.enable_manual_selector_button.config(state=NORMAL)

        except:
            pass

    # --------------------------------------------------------------------------

    def detect_faces(self):
        """
        Runs the face detector and plots the detected faces as red rectangles 
        over the original image.

        Inputs:
            None (self)

        Uses attributes:
            1. self.filename
            2. self.face_detector

        Output:
            1. self.face_detector_results

        Signature:
            face_detector_results = detect_faces()
        """
        # Runs the face detector (detects faces)
        self.face_detector_results = detect_faces(self.filename.name,
                                detector_backend = self.face_detector.get(),
                                align = True, return_type = 'both')

        # Resets all axes (by clearing it) and plots the image
        self.ax.clear()
        self.ax.imshow(self.loaded_img)
        self.ax.set_title(self.filename.name.split("/")[-1])

        # Plots the face detection results as red rectangles overlaid on the
        # original image
        for roi in self.face_detector_results['regions']:
            # Create a Rectangle patch
            rect = Rectangle((roi[0], roi[1]), roi[2], roi[3],
                             linewidth=1, edgecolor='r',
                             facecolor='none')
            
            # Add the patch to the Axes
            self.ax.add_patch(rect)

        # Plots the manual selections as red rectangles overlaid on the original
        # image
        if len(self.manual_coords) > 0:
            for roi in self.manual_coords:
                # Create a Rectangle patch
                rect = Rectangle((roi[0], roi[1]), roi[2], roi[3],
                                 linewidth=1, edgecolor='r',
                                 facecolor='none')
            
                # Add the patch to the Axes
                self.ax.add_patch(rect)

        # Force the axes to update and updates the 'faces' list
        self.chart_type.draw()
        self.update_faces_list([])

    # --------------------------------------------------------------------------

    def update_faces_list(self, selector_coords):
        """
        Updates the 'faces' list, which is the list containing each detected
        face. This list is created from the 'face_detector_results' (which
        contains the faces detected by the face detector) and from the
        'manual_coords' list, which contains the coordinates of manually
        selected faces.

        Inputs:
            1. selector_coords - coordinates from the rectangular selector

        Uses attributes:
            1. self.faces
            2. self.face_detector_results
            3. self.manual_coords
            4. self.loaded_img
            5. self.ax2
            6. self.chart_type2
            7. self.ax_slider
            8. self.result_slider

        Output:
            None (updates the slider and its functionality)
        
        Signature:
            update_faces_list(selector_coords)
        """

        self.faces = [] # initializes the 'faces' list (= "face result list")

        # Populates the 'faces' list with the face detector results (faces)
        if self.face_detector_results != None:
            for face in self.face_detector_results['faces']:
                self.faces.append(face)

        # Update the manual coordinate list with the toggle_selector coordinates
        if len(self.manual_coords) == 0 and len(selector_coords) > 0:
            self.manual_coords = selector_coords
        elif len(selector_coords) > 0:
            for roi in selector_coords:
                if not np.any(np.all(roi == self.manual_coords, axis=1)):
                    self.manual_coords.append(roi)
        
        # Populates the 'faces' list with the faces obtained from the manual
        # coordinates
        if len(self.manual_coords) > 0:
            for roi in self.manual_coords:
                cur_img = self.loaded_img[roi[1]:roi[1]+roi[3],
                                          roi[0]:roi[0]+roi[2], ::-1]
                self.faces.append(cur_img)

        # Always plot the first detected face as 'default'
        self.ax2.imshow(self.faces[0][:, :, ::-1])
        self.ax2.set_title('Detected face 1')
        self.chart_type2.draw()

        # Creates slider (or updates it if it exists) if there are multiple
        # results
        if len(self.faces) - 1 > 0:
            self.result_slider = Slider(self.ax_slider, 'Results', 0,
                                    len(self.faces)-1, valinit=0, valstep=1,
                                    valfmt='%d', orientation='horizontal')

            # Note that update_slider is a function and an object (as Python
            # functions are objects) with attributes
            update_slider.ax = self.ax2
            update_slider.chart_type = self.chart_type2
            update_slider.data = self.faces

            # Calls the 'update_slider' function / object (as a callback) every
            # time the result slider is changed
            self.result_slider.on_changed(update_slider)

    # --------------------------------------------------------------------------

    def enable_manual_selector(self):
        """
        Creates the toggle_selector object (from function with the same name)
        and controls the behaviour of the manual selector button. Also
        initializes a rectangle selector in the toggle_selector object.

        Inputs:
            1. selector_coords - coordinates from the rectangular selector

        Uses attributes:
            1. self.ax
            2. self.chart_type
            3. self.update_faces_list - this is a function
            4. self.enable_manual_selector_button
            5. self.fig
            6. self.manual_coords

        Output:
            None (initializes the interactive rectangle selector)
        
        Signature:
            enable_manual_selector()
        """
        toggle_selector.coords      = []
        toggle_selector.ax_handle   = self.ax
        toggle_selector.chart_type  = self.chart_type
        toggle_selector.update_func = self.update_faces_list
        toggle_selector.button      = self.enable_manual_selector_button

        toggle_selector.RS = RectangleSelector(self.ax, get_cur_coords_callback,
            useblit=True, button=[1, 3],  # don't use middle button
            minspanx=5, minspany=5, spancoords='pixels', interactive=True,
            props=dict(facecolor='red', edgecolor='red', alpha=1,
            fill=False))
        
        self.fig.canvas.mpl_connect('key_press_event', toggle_selector)
        for roi in toggle_selector.coords:
            #print(f'[enable_manual_selector] roi: {roi}')
            self.manual_coords.append(roi)

        #print(f'[enable_manual_selector] manual coords: {self.manual_coords}')
        self.enable_manual_selector_button.config(state=DISABLED)

    # --------------------------------------------------------------------------
    
    def process_faces(self):
        """
        Placeholder: currently not implemented - does nothing!

        Should implement face alignment and other necessary processing that is
        performed inside the 'detect_faces' function. Alignment has been shown
        to improve face recognition performance by at least 1%. This function
        should be used in manually selected faces, as automatically detected
        faces are already aligned.

        Inputs:
            None

        Uses attributes:
            None

        Outputs:
            None

        Signature:
            process_faces()
        """
        pass

    # --------------------------------------------------------------------------


# ==============================================================================
#                                      MAIN
# ==============================================================================
# Starts the face detector app
root=Tk() 
root.configure(bg='white') 
root.title('Face Detector App')
Face_Detector_App(root)
root.resizable(0, 0) 
root.mainloop()
