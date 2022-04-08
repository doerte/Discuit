import sys
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets 
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox

from interface.MainWindow import Ui_MainWindow


class Delegate(QtWidgets.QItemDelegate):
	def __init__(self, owner, choices):
		super().__init__(owner)
		self.items = choices
	def createEditor(self, parent, option, index):
		self.editor = QtWidgets.QComboBox(parent)
		self.editor.addItems(self.items)
		return self.editor
	def paint(self, painter, option, index):
		value = index.data(QtCore.Qt.DisplayRole)
		style = QtWidgets.QApplication.style()
		opt = QtWidgets.QStyleOptionComboBox()
		opt.text = str(value)
		opt.rect = option.rect
		style.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt, painter)
		QtWidgets.QItemDelegate.paint(self, painter, option, index)
	def setEditorData(self, editor, index):
		value = index.data(QtCore.Qt.DisplayRole)
		num = self.items.index(value)
		editor.setCurrentIndex(num)
	def setModelData(self, editor, model, index):
		value = editor.currentText()
		model.setData(index, QtCore.Qt.DisplayRole, QtCore.QVariant(value))
	def updateEditorGeometry(self, editor, option, index):
		editor.setGeometry(option.rect)


class TableModel(QtCore.QAbstractTableModel): 
	def __init__(self, data):
		super().__init__()
		self._data = data
		
	def data(self, index, role): 
		if role == Qt.DisplayRole:
		# See below for the nested-list data structure. # .row() indexes into the outer list,
		# .column() indexes into the sub-list
			value = self._data.iloc[index.row(), index.column()]
			return str(value)

	def setData(self, index, value, role):
		if role == QtCore.Qt.EditRole:
			self._data.iloc[index.row(), index.column()] = value
		return True

	def rowCount(self, index):
		# The length of the outer list.
		return self._data.shape[0]

	def columnCount(self, index):
		# The following takes the first sub-list, and returns
		# the length (only works if all rows are an equal length) 
		return self._data.shape[1]

	def flags(self, index):
		return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

	def headerData(self, section, orientation, role):
		if role == Qt.DisplayRole:
			if orientation == Qt.Horizontal:
				return str(self._data.columns[section])
			#if orientation == Qt.Vertical:
			#	return str(self._data.index[section])


	def isValid( self, fileName ):
		try: 
			file = open( fileName, 'r' )
			file.close()
			return True
		except:
			return False

	def readFile( self, fileName ):
		'''
		sets the member fileName to the value of the argument
		if the file exists.  Otherwise resets both the filename
		and file contents members.
		'''

		if self.isValid( fileName ):
			if fileName.endswith('.csv'):
				self.fileContents = pd.read_csv(fileName)
			else:
				self.fileContents = pd.read_excel(fileName)

		else:
			self.fileContents = ""

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow): 
	def __init__(self):
		super().__init__()
		self.setupUi(self)

		data = pd.DataFrame()
		self.model = TableModel(data)
		self.label.hide()
		self.tableView.hide()

		self.pushButton.clicked.connect(self.browse)
		self.label_2.hide()

	def refreshAll(self, data, variables):
		choices = ['ignore', 'categorical', 'continuous']
		
		#show table view
		self.label.show()
		self.tableView.show()

		#display top 5 (.head()) rows of data file
		self.model= TableModel(data.head())
		self.tableView.setModel(self.model)
		#add combo boxes
		self.tableView.setItemDelegateForRow(0,Delegate(self,choices))
		# make combo boxes editable with a single-click:
		for column in range( self.model._data.shape[1] ):
			self.tableView.openPersistentEditor(self.model.index(0, column))


		self.label_2.show()

	def browse(self):
		#choices = ['ignore', 'categorical', 'continuous']
		fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
						None,
						"Choose File", "",
						"Spreadsheet (*.xlsx *.xls *.csv)")
		if fileName:
			self.model.readFile(fileName)
			data = self.model.fileContents
			

			#add choices on top of data
			variables = list(self.model.fileContents.columns)		
			
			boxes = []
			box = []
			for variable in variables:
				box.append('ignore')
			
			boxes.append(box)
			table = pd.DataFrame(boxes, columns=list(self.model.fileContents.columns))	
			table = pd.concat([table, data], ignore_index=True)

			self.refreshAll(table, variables)



app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()