# -*- coding: utf-8 -*-

# from PyQt5 import QtCore, QtGui, QtWidgets
from DeepTootController import Ui_DeepTootController
import gui_functions
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.ControllerFactory import ControllerFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.BrainFactory import BrainFactory
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


class Ui_DeepTootControllerFull(Ui_DeepTootController):
    """This extension inherits the .ui converted to .py file class and extends it to 
    implement all the connections and signals to complete GUI functionality.
    

    Args:
        Ui_DeepTootController ([type]): [description]
    """    
    def set_event_handling(self):
        self.pushButton_execute.clicked.connect(gui_functions.print_hello)
        
        # Set Driving COntroller combobox update signals
        self.comboBox_drivingControllerType.currentIndexChanged.connect(lambda: self.populate_table(ControllerFactory.create(str(self.comboBox_drivingControllerType.currentText())).params, self.tableWidget_drivingParams))
        self.comboBox_drivingControllerType.currentIndexChanged.connect(lambda: self.populate_table(ControllerFactory.create(str(self.comboBox_drivingControllerType.currentText())).miscOptions, self.tableWidget_drivingMiscOptions))
        
        # Set aerial controller combobox update signals
        self.comboBox_aerialControllerType.currentIndexChanged.connect(
            lambda: self.populate_table(ControllerFactory.create(str(self.comboBox_aerialControllerType.currentText())).params, self.tableWidget_aerialParams)
            )
        self.comboBox_aerialControllerType.currentIndexChanged.connect(
            lambda: self.populate_table(ControllerFactory.create(str(self.comboBox_aerialControllerType.currentText())).miscOptions, self.tableWidget_aerialMiscOptions)
            )
        
        # Set brain presets table combobox signals
        self.comboBox_brainPresets.currentIndexChanged.connect(
            lambda: self.populate_table(BrainFactory.create(str(self.comboBox_brainPresets.currentText())).params, self.tableWidget_brainParams)
        )
        self.comboBox_brainPresets.currentIndexChanged.connect(
            lambda: self.populate_table(BrainFactory.create(str(self.comboBox_brainPresets.currentText())).miscOptions, self.tableWidget_brainMiscOptions)
        )

    def populate_combo_boxes(self):
        """Populate combo boxes with available classes from factories
        """        
        # Driving Controller
        for i in ControllerFactory.controllerList:
            self.comboBox_drivingControllerType.addItem(i)
        
        # Aerial Controller
        for i in ControllerFactory.controllerList:
            self.comboBox_aerialControllerType.addItem(i)
        
        # Brain Presets
        for i in BrainFactory.brainList:
            self.comboBox_brainPresets.addItem(i)

        # Initial COnditions
        # TODO: popultae initial condition combobox

    def populate_table(self, optionList, table: QTableWidget):
        """Take in list of options and table to place options on. Set items on
        QWidgetTable passed in as well

        Args:
            optionList ([dict]): dictionary of parameter names and values
            table (QTableWidget): table to display paramter names and values on
        """        

        # Set table to have 2 columns
        table.setColumnCount(2)
        # Set table to have as many rows as there are options
        table.setRowCount(len(optionList.keys()))

        #Disable sorting
        table.setSortingEnabled(False)

        # remove headers
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)

        for row, key in enumerate(optionList):
            # Set parameter name in first column
            table.setItem(row, 0, QTableWidgetItem(key))
            # Set parameter value in second column
            table.setItem(row, 1, QTableWidgetItem(str(optionList[key])))
        