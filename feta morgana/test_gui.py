import sys
try:
    print("DEBUG: Importing QtWidgets...")
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    print("DEBUG: Importing Matplotlib Backend...")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    print("DEBUG: Creating QApplication...")
    app = QApplication(sys.argv)
    
    print("DEBUG: Creating Window...")
    w = QMainWindow()
    central = QWidget()
    w.setCentralWidget(central)
    layout = QVBoxLayout(central)
    
    print("DEBUG: Creating Figure and Canvas...")
    fig = Figure()
    canvas = FigureCanvasQTAgg(fig)
    layout.addWidget(canvas)
    
    print("DEBUG: Showing Window...")
    w.show()
    
    # Auto-close after 2s for test purposes if running non-interactively
    # But for headless debugging we can just exit
    print("DEBUG: Exiting success...")
    sys.exit(0) # Don't start exec loop to avoid hanging
except BaseException as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
