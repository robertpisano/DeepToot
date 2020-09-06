# -*- coding: utf-8 -*-

# from PyQt5 import QtCore, QtGui, QtWidgets
from DeepTootController import Ui_DeepTootController
import gui_functions
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.ControllerFactory import ControllerFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.BrainFactory import BrainFactory
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


class Ui_DeepTootControllerFull(Ui_DeepTootController):
    """This extension takes the .ui converted to .py file class and extends it to 
    implement all the connections and signals to make the GUI work properly.
    

    Args:
        Ui_DeepTootController ([type]): [description]
    """    
    def set_event_handling(self):
        self.pushButton_execute.clicked.connect(gui_functions.print_hello)
        
        # Set Driving COntroller table update signals
        self.comboBox_drivingControllerType.currentIndexChanged.connect(lambda: self.populate_table(ControllerFactory.create(str(self.comboBox_drivingControllerType.currentText())).params, self.tableWidget_drivingParams))
        self.comboBox_drivingControllerType.currentIndexChanged.connect(lambda: self.populate_table(ControllerFactory.create(str(self.comboBox_drivingControllerType.currentText())).miscOptions, self.tableWidget_drivingMiscOptions))
        
        # Set aerial controller table update signals
        self.comboBox_aerialControllerType.currentIndexChanged.connect(lambda: self.populate_table(ControllerFactory.create(str(self.comboBox_aerialControllerType.currentText())).params, self.tableWidget_aerialParams))
        self.comboBox_aerialControllerType.currentIndexChanged.connect(lambda: self.populate_table(ControllerFactory.create(str(self.comboBox_aerialControllerType.currentText())).miscOptions, self.tableWidget_aerialMiscOptions))
        
        # Set brain presets table update signals
        self.comboBox_brainPresets.currentIndexChanged.connect(
            lambda: self.populate_table(BrainFactory.create(str(self.comboBox_brainPresets.currentText())).params, self.tableWidget_brainParams)
        )
        self.comboBox_brainPresets.currentIndexChanged.connect(
            lambda: self.populate_table(BrainFactory.create(str(self.comboBox_brainPresets.currentText())).miscOptions, self.tableWidget_brainMiscOptions)
        )

    def populate_combo_boxes(self):
        """Populate combo boxes with available classes
        """        
        for i in ControllerFactory.controllerList:
            self.comboBox_drivingControllerType.addItem(i)
        
        for i in ControllerFactory.controllerList:
            self.comboBox_aerialControllerType.addItem(i)
        
        for i in BrainFactory.brainList:
            self.comboBox_brainPresets.addItem(i)

    def populate_table(self, optionList, table: QTableWidget):
        table.setColumnCount(2)
        table.setRowCount(len(optionList.keys()))

        table.setSortingEnabled(False)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)

        for row, key in enumerate(optionList):
            table.setItem(row, 0, QTableWidgetItem(key))
            table.setItem(row, 1, QTableWidgetItem(str(optionList[key])))
        