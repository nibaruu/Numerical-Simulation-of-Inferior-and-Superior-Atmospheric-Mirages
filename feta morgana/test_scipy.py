import sys
try:
    print("DEBUG: Importing Scipy...")
    import scipy.ndimage
    print("DEBUG: Scipy OK")
except ImportError as e:
    print(f"DEBUG: Scipy FAILED: {e}")
    sys.exit(1)
except BaseException as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
