# ==============================================================================
#                                GLOBAL VARIABLES
# ==============================================================================

models      = []     # stores all face verifier models
rep_db      = []     # stores representation database
db_changed  = False  # indicates if database has been modified and should be changed
faiss_index = []     # stores faiss index for verification speed up
