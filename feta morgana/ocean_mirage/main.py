import sys


def main() -> None:
    try:
        from PyQt6.QtWidgets import QApplication
        from ui import OceanMirageWindow

        app = QApplication(sys.argv)
        app.setStyle("Fusion")

        window = OceanMirageWindow()
        window.show()

        sys.exit(app.exec())
    except Exception as exc:
        import traceback
        with open("ocean_crash.log", "w") as f:
            f.write(traceback.format_exc())
        print(f"CRASH: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
