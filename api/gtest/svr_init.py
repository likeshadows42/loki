# ==============================================================================
#                             SERVER CONFIGURATION
# ==============================================================================

from multiprocessing        import Manager

# ______________________________________________________________________________
#                                GLOBAL VARIABLES
# ------------------------------------------------------------------------------

manager_dict = {}

# ______________________________________________________________________________
#                                 SERVER METHODS
# ------------------------------------------------------------------------------

def initialization():
    DEBUG = True  # this should be loaded from a config file perhaps?

    if DEBUG:
        print('\n ======== Starting initialization process ======== \n')

    # Creating multiprocessing manager
    global manager_dict
    manager_dict = Manager().dict()                 # Manager

    # Random values / objects
    manager_dict['val1'] = 100
    manager_dict['val2'] = 'abc'
    manager_dict['val3'] = None

    if DEBUG:
        print('\n -------- End of initialization process -------- \n')

# ------------------------------------------------------------------------------

def when_ready(server):
    initialization()

# ------------------------------------------------------------------------------
