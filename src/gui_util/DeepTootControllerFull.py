# -*- coding: utf-8 -*-

# from PyQt5 import QtCore, QtGui, QtWidgets
from DeepTootController import Ui_DeepTootController
import gui_functions
from DeepToot.src.meta_data_objects.controllers.Controller import Controller
from DeepToot.src.meta_data_objects.controllers.ControllerFactory import ControllerFactory, ControllerSchema
from DeepToot.src.meta_data_objects.brains.BrainFactory import BrainFactory, BrainSchema
from DeepToot.src.meta_data_objects.InitialConditions.InitialConditionsFactory import InitialConditionsFactory, InitialConditionsSchema
from DeepToot.src.meta_data_objects.AbstractMetaDataObjectFactory import AbstractMetaDataObjectFactory
from DeepToot.src.meta_data_objects.MetaDataObject import MetaDataObject
from DeepToot.src.meta_data_objects.SimulationDataObject import SimulationDataObject, SimulationDataObjectSchema
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractScrollArea

# Socket imports
from DeepToot.src.comms_util.client import Client
from DeepToot.src.comms_util.comms_protocol import CommsProtocol

class Ui_DeepTootControllerFull(Ui_DeepTootController):
    """This extension inherits the .ui converted to .py file class and extends it to 
    implement all the connections and signals to complete GUI functionality.
    

    Args:
        Ui_DeepTootController ([type]): [description]
    """    
    def set_event_handling(self):
        self.pushButton_execute.clicked.connect(self.execute)
        self.pushButton_update.clicked.connect(self.update_classes)
        self.pushButton_terminate.clicked.connect(self.terminate)

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
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_initialConditionsPresets.currentText())).params, self.tableWidget_initialConditionsParameters)
            )
        
        self.comboBox_initialConditionsPresets.currentIndexChanged.connect(
            lambda: self.populate_table(AbstractMetaDataObjectFactory.create(str(self.comboBox_initialConditionsPresets.currentText())).miscOptions, self.tableWidget_initialConditionsMiscOptions)
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
        for i in AbstractMetaDataObjectFactory.initialConditionsList:
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
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

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

        table.resizeColumnsToContents()
    
    def generate_data_object_from_table(self, comboBox, tableParams:QTableWidget, tableMiscOptions:QTableWidget):
        # Get type from combobox
        obj = AbstractMetaDataObjectFactory.create(str(comboBox.currentText()))
        
        for row in range(0, tableParams.rowCount()):
            obj.params[tableParams.item(row, 0).text()] = tableParams.item(row, 1).text()
            print('table data: ', tableParams.item(row, 0).text(), tableParams.item(row,1).text())
            print('added param: ', obj.params[tableParams.item(row, 0).text()])
        print('obj.params', obj.params)

        for row in range(0, tableMiscOptions.rowCount()):
            obj.miscOptions[tableMiscOptions.item(row, 0).text()] = tableMiscOptions.item(row, 1).text()
            print('added miscOptions: ', obj.miscOptions[tableMiscOptions.item(row, 0).text()])

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
        ic = self.generate_data_object_from_table(self.comboBox_initialConditionsPresets, self.tableWidget_initialConditionsParameters, self.tableWidget_initialConditionsMiscOptions)

        # Generate simulation data object
        simulationDataObject = SimulationDataObject(dc, ac, b, ic)


        # # Serialize object
        serialized = SimulationDataObjectSchema().dumps(simulationDataObject)
        print(serialized)
        # Send object over socket
        c = Client()
        c.send_message(CommsProtocol.types['execute'], serialized)
    
    def update_classes(self):
        c = Client()
        c.send_message(CommsProtocol.types['update'], 'empty')
    
    def terminate(self):
        c = Client()
        c.send_message(CommsProtocol.types['terminate'], 'empty')