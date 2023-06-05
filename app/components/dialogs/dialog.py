# coding:utf-8
from app.common.auto_wrap import autoWrap
from PyQt5.QtCore import Qt, pyqtSignal, QFile
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton


class Dialog(QDialog):

    yesSignal = pyqtSignal()
    cancelSignal = pyqtSignal()

    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent, Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.resize(300, 200)
        self.setWindowTitle(title)
        self.content = content
        self.titleLabel = QLabel(title, self)
        self.contentLabel = QLabel(content, self)
        self.yesButton = QPushButton(self.tr('OK'), self)
        self.cancelButton = QPushButton(self.tr('Cancel'), self)
        self.__initWidget()

    def __initWidget(self):
        """ 初始化小部件 """
        self.yesButton.setFocus()
        self.titleLabel.move(30, 22)
        # self.contentLabel.setMaximumWidth(900)
        self.contentLabel.setText(autoWrap(self.content, 100)[0])
        self.contentLabel.move(30, self.titleLabel.y()+50)

        # 设置层叠样式
        self.__setQss()

        # 调整窗口大小
        rect = self.contentLabel.rect()
        self.setFixedSize(60+rect.right()+self.cancelButton.width(),
                          self.contentLabel.y()+self.contentLabel.height()+self.yesButton.height()+60)

        # 信号连接到槽
        self.yesButton.clicked.connect(self.__onYesButtonClicked)
        self.cancelButton.clicked.connect(self.__onCancelButtonClicked)

    def resizeEvent(self, e):
        self.cancelButton.move(self.width()-self.cancelButton.width()-30,
                               self.height()-self.cancelButton.height()-30)
        self.yesButton.move(self.cancelButton.x() -
                            self.yesButton.width()-30, self.cancelButton.y())

    def __onCancelButtonClicked(self):
        self.cancelSignal.emit()
        self.deleteLater()

    def __onYesButtonClicked(self):
        self.yesSignal.emit()
        self.deleteLater()

    def __setQss(self):
        """ 设置层叠样式 """
        self.titleLabel.setObjectName("titleLabel")
        self.contentLabel.setObjectName("contentLabel")

        f = QFile(':/qss/dialog.qss')
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()

        self.yesButton.adjustSize()
        self.cancelButton.adjustSize()
        self.contentLabel.adjustSize()
