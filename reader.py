import json
import os
import shutil
import sys
import traceback
from collections import Counter

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSlot, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QBrush
from PyQt5.QtWidgets import QApplication, QFileDialog, QTableView, QMenu

from config import *
from gears.detector import detector
from gears.extractor import value_extractor
from gears.helpers import export_json


class GearTable(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(GearTable, self).__init__()

        header = ['img', 'rating', 'level', 'rarity', 'set',
                  'slot', 'ability', 'main', 'value',
                  "Atk", "AtkP", "CChance", "CDmg", "Def",
                  "DefP", "Eff", "HP", "HPP", "Res", "Spd",
                  ]

        self.header = header
        self.data = data
        self.clean_data()
        # self.col_name = [key for key, val in data[0].items()]

    def data(self, index, role):
        try:
            # key = self.header[index.column()]
            # if "mainStat" in key:
            #     value = self.data[index.row()][key]
            # else:
            value = self.data[index.row()][self.header[index.column()]]
        except:
            value = ""

        if role == Qt.DisplayRole or role == Qt.EditRole:
            return value
        if role == Qt.BackgroundRole and (value in ['999', 'NA', 999, 0]):
            return QBrush(Qt.red)

    def rowCount(self, index):
        # The length of the outer list.
        return len(self.data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        # len(self._data[0])
        return len(self.header)

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.header[section])
            if orientation == Qt.Vertical:
                return str(section)

    def setData(self, index, value, role=Qt.EditRole):
        # old_val = self.data[index.row()][self.header[index.column()]]
        # print(old_val)
        if value != "":  # and value != old_val:
            self.data[index.row()][self.header[index.column()]] = value
            self.data[index.row()]['rating'] = self.rate_gear(index)
            self.dataChanged.emit(index, index)
            return True
        return False

    def rate_gear(self, index):
        stats_weight = [("Atk", 0.01), ("AtkP", 1), ("CChance", 1.5), ("CDmg", 1), ("Def", 0),
                        ("DefP", 1), ("Eff", 1), ("HP", 0.002), ("HPP", 1), ("Res", 1), ("Spd", 2)]
        rating = 0
        print(self.data[index.row()].items())
        for key, val in self.data[index.row()].items():
            for stat, w in stats_weight:
                if key == stat:
                    print(key, val)
                    rating += int(val) * w
        print("Done")
        return int(rating)

    def flags(self, index):
        return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def clean_data(self):
        for i, row in enumerate(self.data):
            for key in list(row):
                if key not in self.header:
                    if "Stat" in key:
                        val = row[key]
                        if "main" in key:
                            self.data[i]['main'] = val[0]
                            self.data[i]['value'] = int(val[1])
                        else:
                            self.data[i][val[0]] = int(val[1])
                    del self.data[i][key]
        print("cleaned")

    def remove_row(self, position, rows=1):
        self.beginRemoveRows(QtCore.QModelIndex(), position, position + rows - 1)
        self.data = self.data[:position] + self.data[position + rows:]
        self.endRemoveRows()
        print("deleted")
        return True


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()  # Call the inherited classes __init__ method
        uic.loadUi('main.ui', self)  # Load the .ui file
        self.setWindowTitle("E7 Gear Reader")
        self.threadpool = QThreadPool()

        # Load widgets

        self.run_btn.clicked.connect(self.reader)
        self.select_btn.clicked.connect(self.select_file)
        self.export_btn.clicked.connect(self.export)
        self.progress_bar.setValue(0)
        self.clear_btn.clicked.connect(self.clear_db)
        self.end_task_btn.clicked.connect(self.end_task)

        self.actionWeb_version.triggered.connect(lambda: self.openUrl(url="https://e7gears.herokuapp.com"))
        self.actionDonate.triggered.connect(lambda: self.openUrl(url="https://www.buymeacoffee.com/e7gears"))
        self.actionGithub.triggered.connect(lambda: self.openUrl(url="https://github.com/e7-tools/gear-reader"))

        #
        self.load_table()

    def openUrl(self, url):
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)):
            QtGui.QMessageBox.warning(self, 'Open Url', 'Could not open url')

    def end_task(self):
        self.task = False

    def clear_db(self):
        shutil.rmtree(IMAGES_DIR)
        self.model.data = []
        self.save()
        self.model.layoutChanged.emit()
        self.img_box.setText("No image available.")

    def remove_row(self):
        r = self.table.currentIndex().row()

        # Delete img
        img_file = self.table.model().data[r]['img']
        if os.path.exists(img_file):
            os.remove(img_file)
        # Delete row
        self.table.model().remove_row(r)
        self.save()
        print("Deleted row")

    def select_file(self):
        self.file, _ = QFileDialog.getOpenFileName(self, "Select video to read", "",
                                                   "MP4 File (*.mp4)")
        self.input_label.setText(os.path.basename(self.file))

    def export(self):
        with open(DATABASE, 'r') as f:
            data = json.load(f)

        my_inventory = export_json(data)

        filename, ok = QFileDialog.getSaveFileName(self, 'Save File', "",
                                                   "Json File (*.json)")
        if ok and filename:
            with open(filename, 'w') as f:
                json.dump(my_inventory, f)
            print("Exported")

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)
        delete = contextMenu.addAction("Delete gear")
        action = contextMenu.exec_(self.mapToGlobal(event.pos()))
        if action == delete:
            self.remove_row()

    def show_img(self, index):
        pixmap = QPixmap(index.sibling(index.row(), 0).data())
        print(index.sibling(index.row(), 0).data())
        self.img_box.setPixmap(pixmap)

    def editor(self, index):
        selected_col = self.model.header[index.column()]
        columns = ['rarity', 'set', 'slot', 'main']
        options = [("Epic", "Heroic", "Rare"),
                   ('Speed', 'Critical', 'Hit', 'Attack', 'Defense', 'Health', 'Destruction',
                    'Immunity', 'Lifesteal', 'Rage', 'Resist', 'Unity', 'Counter'),
                   ('Weapon', 'Helmet', 'Armor', 'Necklace', 'Ring', 'Boots'),
                   ("Atk", "AtkP", "CChance", "CDmg", "Def",
                    "DefP", "Eff", "HP", "HPP", "Res", "Spd")
                   ]
        for idx, col in enumerate(columns):
            if selected_col == col:
                self.editor_helper(index, col, options[idx])

    def editor_helper(self, index, column, options):
        item, ok = QtWidgets.QInputDialog.getItem(self, "Select input dialog",
                                                  f"List of {column} options:", options, 0, False)
        if ok and item:
            print(self.model.data[index.row()][column])
            self.model.data[index.row()][column] = item
            print("ok", item)

    def load_table(self):
        # Open saved data if any
        try:
            self.table = self.findChild(QtWidgets.QTableView, 'gearTable')
            if os.path.exists(DATABASE):
                with open(DATABASE, 'r') as f:
                    self.db = json.load(f)
                    print(self.db)
            else:
                self.db = []
            self.model = GearTable(self.db)
            self.table.setModel(self.model)
            self.table.clicked.connect(self.show_img)
            self.table.doubleClicked.connect(self.editor)
            # self.installEventFilter(self.table)

            self.table.setSelectionBehavior(QTableView.SelectRows)
            self.table.setColumnHidden(0, True)
            self.table.resizeColumnsToContents()
            self.table.model().dataChanged.connect(self.save)

        except Exception:
            self.db = []
            print("No data")
            pass

    def save(self):
        with open(DATABASE, 'w') as f:
            data = json.dump(self.model.data, f)

    ##### Threading
    def progress_fn(self, gear, errors, count, total):
        # print(f"Gear Progress {count} over {total}.\nNew gear {gear}")
        if gear == "starting":
            self.status.setText(errors)
        elif gear == "detecting":
            self.status.setText(f"Capturing gear frames from video.\nFound {count} frames.")
        elif gear == "extracting":
            self.status.setText(f"Found {total} frames. Start reading gear data from frames.")
        else:
            percent = int(count * 100 / total)
            self.model.data.append(gear)
            self.model.layoutChanged.emit()
            self.save()
            self.status.setText(f"Reading gear data from {count + 1}th over {total} frames.\n{errors}")
            self.progress_bar.setValue(percent)

    def gear_fn(self, file, progress_callback):
        print("Thread start")
        if TESSERACT_DIR is None:
            return "Tesseract-OCR not found.\nPlease install and setup path."
        else:
            progress_callback.emit("starting", "Starting.....", None, None)

        if not os.path.exists(IMAGES_DIR):
            os.mkdir(IMAGES_DIR)

        if ".mp4" in file:
            img_files = detector(file, IMAGES_DIR, progress_callback)
        else:
            import random
            img_files = [os.path.join(r, f) for r, d, fd in os.walk(file) for f in fd]
            random.shuffle(img_files)
        if len(img_files) == 0:
            return "No gear found. Please check your video."

        count = 0
        errors = Counter()
        total = len(img_files)

        progress_callback.emit("extracting", None, None, total)

        for img in img_files:
            print(img)

            if not self.task:
                return f"Task stopped."
            # print(img)
            try:
                gear, error = value_extractor(img, DEBUG)
                print("read gear")
                gear['img'] = img.replace(".jpg", "-box.jpg")
                errors += Counter(error)

            except Exception as e:
                errors['gear'] += 1
                gear = {"img": img}
                print("Error:", e)

            count += 1
            print_errs = "Errors count:\n"

            for e, ev in dict(errors).items():
                print_errs += f"{e}: {ev}, "
            print(print_errs)

            progress_callback.emit(gear, print_errs, count, total)

        return f"Finished importing {total} gears.\n{print_errs}"

    def print_output(self, result):
        self.status.setText(result)

    def complete_fn(self):
        self.table.resizeColumnsToContents()
        print("THREAD COMPLETE!")

    def reader(self):
        if self.file:
            # Pass the function to execute
            self.task = True
            worker = Worker(self.gear_fn, file=self.file)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.print_output)
            worker.signals.finished.connect(self.complete_fn)
            worker.signals.progress.connect(self.progress_fn)

            # Execute
            self.threadpool.start(worker)
            self.file = None
            self.input_label.setText("No file selected.")


### Threads

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            print((exctype, value, traceback.format_exc()))
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(object, object, int, int)


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('./templates/icon.ico'))
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
