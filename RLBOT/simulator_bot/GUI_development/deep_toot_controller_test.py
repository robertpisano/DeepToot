import sys
from PyQt5 import QtWidgets, uic

from DeepTootControllerFull import Ui_DeepTootControllerFull
from socket_types import ScenarioData


class MainWindow(QtWidgets.QMainWindow, Ui_DeepTootControllerFull):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.set_event_handling()
        self.populate_combo_boxes()
        


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()