# -*- coding: utf-8 -*-

# from PyQt5 import QtCore, QtGui, QtWidgets
from DeepTootController import Ui_DeepTootController
import gui_functions
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.Controller import Controller
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.controllers.ControllerFactory import ControllerFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.brains.BrainFactory import BrainFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.InitialConditions.InitialConditionsFactory import InitialConditionsFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.AbstractMetaDataObjectFactory import AbstractMetaDataObjectFactory
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.MetaDataObject import MetaDataObject
from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.SimulationDataObject import SimulationDataObject
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

# Socket imports
from DeepToot.RLBOT.simulator_bot.GUI_development import client

class Ui_DeepTootControllerFull(Ui_DeepTootController):
    """This extension inherits the .ui converted to .py file class and extends it to 
    implement all the connections and signals to complete GUI functionality.
    

    Args:
        Ui_DeepTootController ([type]): [description]
    """    
    def set_event_handling(self):
        self.pushButton_execute.clicked.connect(self.execute)
        
        # Set Driving COntroller combobox update signals
        self.comboBox_drivingControllerType.currentIndexChanged.connect(
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_drivingControllerType.currentText())).params, self.tableWidget_drivingParams)
            )
        self.comboBox_drivingControllerType.currentIndexChanged.connect(
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_drivingControllerType.currentText())).miscOptions, self.tableWidget_drivingMiscOptions)
            )

        
        # Set aerial controller combobox update signals
        self.comboBox_aerialControllerType.currentIndexChanged.connect(
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_aerialControllerType.currentText())).params, self.tableWidget_aerialParams)
            )
        self.comboBox_aerialControllerType.currentIndexChanged.connect(
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_aerialControllerType.currentText())).miscOptions, self.tableWidget_aerialMiscOptions)
            )
        
        # Set brain presets table combobox signals
        self.comboBox_brainPresets.currentIndexChanged.connect(
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_brainPresets.currentText())).params, self.tableWidget_brainParams)
            )
        self.comboBox_brainPresets.currentIndexChanged.connect(
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_brainPresets.currentText())).miscOptions, self.tableWidget_brainMiscOptions)
            )

        # Set Initial conditions table combobox signals
        self.comboBox_initialConditionsPresets.currentIndexChanged.connect(
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_initialConditionsPresets.currentText())).params, self.tableWidget_initialConditions)
            )

    def populate_combo_boxes(self):
        """Populate combo boxes with available classes from factories
        """        
        # Driving Controller
        for i in ControllerFactory.list:
            self.comboBox_drivingControllerType.addItem(i)

        # Aerial Controller
        for i in ControllerFactory.list:
            self.comboBox_aerialControllerType.addItem(i)
        
        # Brain Presets
        for i in BrainFactory.list:
            self.comboBox_brainPresets.addItem(i)

        # Initial COnditions
        for i in InitialConditionsFactory.list:
            self.comboBox_initialConditionsPresets.addItem(i)

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
    
    def generate_data_object_from_table(self, comboBox, tableParams:QTableWidget, tableMiscOptions:QTableWidget):
        # Get type from combobox
        obj = AbstractMetaDataObjectFactory.create(str(comboBox.currentText()))
        
        for row in range(0, tableParams.rowCount()):
            obj.params[tableParams.item(row, 0).text()] = tableParams.item(row, 1).text()
            print('table data: ', tableParams.item(row, 0).text(), tableParams.item(row,1).text())
            print('obj.params', obj.params)

        for row in range(0, tableMiscOptions.rowCount()):
            obj.miscOptions[tableMiscOptions.item(row, 0).text()] = tableMiscOptions.item(row, 1).text()
            print(tableMiscOptions.item(row, 0).text())

        return obj
        
        
    def execute(self):
        """Execute will parse data from the tables and intantiate a SimulationDataObject to be sent over the socket.
        It will then be sent over the socket to the bot.
        """        
        # Instantiate driving controller
        dc = self.generate_data_object_from_table(self.comboBox_drivingControllerType, self.tableWidget_drivingParams, self.tableWidget_drivingMiscOptions)
        
        # aerial controller
        ac = self.generate_data_object_from_table(self.comboBox_aerialControllerType, self.tableWidget_aerialParams, self.tableWidget_aerialMiscOptions)
        
        # brain
        b = self.generate_data_object_from_table(self.comboBox_brainPresets, self.tableWidget_brainParams, self.tableWidget_brainMiscOptions)
        
        # initial conditions
        ic = self.generate_data_object_from_table(self.comboBox_initialConditionsPresets, self.tableWidget_initialConditions, self.tableWidget_initialConditions)

        # Generate simulation data object
        simulationDataObject = SimulationDataObject(dc, ac, b, ic)


        # Permenantly change dc.param to see if change is emulated when object is recreated from serialized data
        # simulationDataObject.drivingController = AbstractMetaDataObjectFactory.create('DrivingController')
        # simulationDataObject.drivingController.params['kp'] = 101010
        # print(simulationDataObject.drivingController.params)

        # # DEBUGGING:
        from DeepToot.RLBOT.simulator_bot.GUI_development.meta_data_objects.SerializationFactory import SerializationFactory
        # # Serialize object
        serialized = SerializationFactory.listify(simulationDataObject)

        deserialized = SerializationFactory.delistify(serialized)

        print(deserialized)
        # print()
        # pass


        # Send object over socket
        client.sendSocketMessage(simulationDataObject)