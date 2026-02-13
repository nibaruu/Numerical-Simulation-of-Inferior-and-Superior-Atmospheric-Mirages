import sys
try:
    print("Importing PyQt6...")
    from PyQt6.QtWidgets import QApplication
    print("PyQt6 OK")
except ImportError as e:
    print(f"PyQt6 FAILED: {e}")

try:
    print("Importing Matplotlib...")
    import matplotlib.pyplot as plt
    print("Matplotlib OK")
except ImportError as e:
    print(f"Matplotlib FAILED: {e}")

try:
    print("Importing FigureCanvasQTAgg...")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    print("Backend OK")
except ImportError as e:
    print(f"Backend FAILED: {e}")
