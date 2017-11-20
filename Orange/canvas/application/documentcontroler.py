import os
import sys
import time
from collections import namedtuple

try:
    from typing import List, Optional, Type
except ImportError:
    pass


from AnyQt.QtWidgets import (
    QAction, QActionGroup, QMenu, QFileDialog, QMessageBox, QApplication,
    QFileIconProvider
)
from AnyQt.QtGui import QKeySequence
from AnyQt.QtCore import (
    Qt, QEvent, QObject, QUrl, QStandardPaths, QFileInfo, QSettings,
    QMimeDatabase, QMimeType
)
from AnyQt.QtCore import pyqtSignal as Signal


from Orange.canvas.gui import utils


class Document(QObject):
    """
    An abstract representation of a open document in an application.

    This class is used by the document controller open, save and keep
    track of documents in an application. It is also responsible for
    the creation/definition of GUI elements associated with a document
    type.
    """

    class Type(namedtuple("Type", ["name", "mimeType", "suffixes"])):
        pass

    #: Url associated with the document has changed.
    pathChanged = Signal(QUrl)
    #: Document title/name changed
    titleChanged = Signal(str)

    #: Document meta properties changed
    propertiesChanged = Signal(str)

    #: Widget for document editing/display has been created.
    viewCreated = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.__url = None
        self.__title = ""
        self.__transient = False
        self.__modified = False
        self.__documentController = None

        self.__action_close = QAction(
            self.tr("Close"), self,
            objectName="action-close",
            triggered=self.close,
            shortcut=QKeySequence.Close
        )

        self.__action_close_window = QAction(
            self.tr("Close Window"), self,
            objectName="action-close-view",
            triggered=self.closeView,
            shortcut=Qt.ShiftModifier | QKeySequence.Close
        )

        self.__action_save = QAction(
            self.tr("Save"), self,
            objectName="action-save",
            triggered=self.save,
            shortcut=QKeySequence.Save
        )

        self.__action_save_as = QAction(
            self.tr("Save As"), self,
            objectName="action-save-as",
            triggered=self.saveAs,
            shortcut=QKeySequence.SaveAs
        )

    @classmethod
    def supportedMimeTypes(cls):
        """
        Return the supported document types

        Returns
        -------
        mimetypes : List[str]
        """
        raise NotImplementedError

    def widget(self):
        """
        Return a widget/view of the open document.

        Returns
        -------
        widget : QWidget
        """
        raise NotImplementedError

    def documentController(self):
        """
        Return the DocumentController instance which manages this document.

        Return None if the document is not associated with a controller

        Returns
        -------
        controller : Optional[DocumentController]
        """
        return self.__documentController

    def _setDocumentController(self, controller):
        """
        NOTE: Only the document controller should call this
        """
        self.__documentController = controller

    def path(self):
        """
        Return the url path associated with this document.
        """
        return QUrl(self.__url)
    url = path

    def setPath(self, url):
        """
        Associate an url with this document.
        """
        url = QUrl(url)
        if self.__url != url:
            self.__url = url
            self.pathChanged.emit(url)
    setUrl = setPath

    def setTitle(self, title):
        """
        Set the document title (display name).
        """
        if self.__title != title:
            self.__title = title
            self.titleChanged.emit(title)

    def title(self):
        """
        Return the document title.
        """
        return self.__title

    def read(self, url, doctype=None):
        # type: (QUrl, Optional[Document.Type]) -> None
        """
        Read and load a document from `url`.
        """
        raise NotImplementedError

    def write(self, url, doctype=None):
        """
        Save the document to an associated url.

        If no associated url is set a user is presented with a file
        dialog to select a file system path.

        This is the slot associated with the :gui:`Save` menu action.

        """
        if self.url():
            return self.write_to(self.url())
        else:
            return self.save_as()

    def save(self):
        if self.url():
            self.saveToPath(self.url())
        else:
            self.saveAs()

    def saveAs(self):
        filename, fileformat = self.run_save_file_dialog()
        if filename:
            return self.write_to(filename, fileformat)
        else:
            return False

    def saveToPath(self, url):
        try:
            sucess = self.write(url, self.documentType())
        except Exception:  # pylint: disable=broad-except
            self.report_save_error(url)
            return False

        if sucess:
            self.setPath(url)
        else:
            return True

    def writeToPath(self, url, doctype=None):
        raise NotImplementedError

    def isModified(self):
        return self.__modified

    def setModified(self, modified):
        if self.__modified != modified:
            self.__modified = bool(modified)
            self.modifiedChanged.emit(self.__modified)

    def isTransient(self):
        return self.__transient

    def close(self):
        if self.isModified() and not self.isTransient():
            title = self.title()
            url = self.path()
            if not title:
                if url:
                    title = os.path.basename(url)
                else:
                    title = "untitled"

            result = utils.message_question(
                self.tr("Do you want to save the changes you made "
                        "to document '%s'?" % title),
                self.tr("Save Changes?"),
                self.tr("Your changes will be lost if you do not save them."),
                buttons=(QMessageBox.Save | QMessageBox.Cancel |
                         QMessageBox.Discard),
                default_button=QMessageBox.Save,
                parent=self.widget()
            )
            if result == QMessageBox.Save:
                return self.save()
            elif result == QMessageBox.Discard:
                return True
            elif result == QMessageBox.Cancel:
                return False
        else:
            return True

    def closeView(self):
        """
        Attempt to close the associated widget/view.

        Return a boolean indicating if the widget accepted the
        close request.

        Returns
        -------
        accepted : bool
        """
        return self.widget().close()

    def activate(self):
        """
        Activate (show, raise and give focus) the associated widget/view.
        """
        widget = self.widget()
        if widget:
            widget = self.widget()
            widget.show()
            widget.raise_()
            widget.activateWindow()

    def saveFileDialog(self, ):
        ## This should be controllers/application's responsibility
        dialog = QFileDialog(
            self.widget(),
            fileMode=QFileDialog.ExistingFile,
            acceptMode=QFileDialog.AcceptSave,
            windowTitle=self.tr("Save"))
        dialog.setDefaultSuffix(self.documentType().ext)
        directory = QStandardPaths.writableLocation(
            QStandardPaths.DocumentsLocation)
        if directory:
            dialog.setDirectory(directory)
        types = self.documentType()
        spec = types_to_filters(types)
        dialog.setNameFilters(spec)
        # set current filter
        return dialog

    def runSaveFileDialog(self, ):
        dialog = self.saveFileDialog()
        dialog.show()
        # dialog.done.connect(...)

    def undoStack(self):
        """
        Return a QUndoStack for the document.

        The default implementation returns None.
        """
        return None

    def eventFilter(self, receiver, event):
        if event.type() == QEvent.Close and receiver is self.widget():
            event.setAccepted(self.close())
            return True
        else:
            return super().eventFilter(receiver, event)


class TextDocument(Document):
    """
    A plain text document
    """
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__contents = ""
        self.__widget = None

    @classmethod
    def supportedMimeTypes(cls):
        return ["text/plain"]

    def read(self, url, doctype=None):
        filepath = url.toLocalFile()
        try:
            with open(filepath, "r") as f:
                txt = f.read()
        except (IOError, OSError) as err:
            utils.message_warning(
                self.tr("Could not open '{}'.".format(filepath)),
                title=self.tr("Error"),
                informative_text=os.strerror(err.errno)
            )
            return False
        else:
            self.__contents = txt
            return True

    def writeToPath(self, url, doctype=None):
        filepath = url.toLocalFile()
        try:
            with open(filepath, ) as f:
                f.write(self.__contents)
        except OSError:
            ...

    def widget(self):
        if self.__widget is None:
            from AnyQt.QtWidgets import QPlainTextEdit
            w = QPlainTextEdit()
            if self.__contents is not None:
                w.setPlainText(self.__contents)
            self.__widget = w
        return self.__widget


class SvgDocument(Document):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__contents = ""
        self.__widget = None

    @classmethod
    def supportedMimeTypes(cls):
        return ["text/svg"]

    def read(self, url, doctype=None):
        filepath = url.toLocalFile()
        try:
            with open(filepath, "rb") as f:
                content = f.read()
        except OSError:
            ...
        else:
            self.__contents = content
            return True

    def writeToPath(self, url, doctype=None):
        filepath = url.toLocalFile()
        try:
            with open(filepath, "rb") as f:
                f.write(self.__contents)
        except OSError as err:
            ...

    def widget(self):
        if self.__widget is None:
            from PyQt5.QtSvg import QSvgWidget
            w = QSvgWidget(self.__contents)
            self.__widget = w
        return self.__widget


from AnyQt.QtGui import QImageReader, QImage, QPixmap


class ImageDocument(Document):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__contents = QImage()
        self.__widget = None

    @classmethod
    def supportedMimeTypes(cls):

        return [bytes(t).decode("utf-8") for t in QImageReader.supportedMimeTypes()]

    def read(self, url, doctype=None):
        filepath = url.toLocalFile()
        try:
            self.__contents = QImage(filepath)
            return True
        except OSError:
            ...
        return False

    def widget(self):
        if self.__widget is None:
            from PyQt5.QtWidgets import QLabel
            w = QLabel()
            w.setPixmap(QPixmap.fromImage(self.__contents))
            self.__widget = w
        return self.__widget


class DocumentController(QObject):
    """
    A controller/manager of open documents in an application.

    """
    documentOpened = Signal(Document)
    documentClosed = Signal(Document)
    #: Document's modified state has changed
    documentModifiedChanged = Signal(Document)
    #: The current topmost open document has changed
    currentDocumentChanged = Signal(Document)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__maxRecentCount = 10
        self.__lastDirectory = ""
        self.__documentTypes = []

        self.__defaultDocumentClass = None
        #: All open documents.
        self.__documents = []  # type: List[Document]

        self.__action_new = QAction(
            self.tr("New"), self,
            objectName="action-new",
            triggered=self.newDocument,
            shortcut=QKeySequence.New,
        )

        self.__action_open = QAction(
            self.tr("Open"), self,
            objectName="action-open",
            triggered=self.open,
            shortcut=QKeySequence.Open
        )

        self.__recent_group = QActionGroup(self, exclusive=False)

        self.__action_recent = QAction(
            self.tr("Open Recent"), self,
            objectName="action-open-recent",
        )

        self.__action_clear_recent = QAction(
            self.tr("Clear Recent"), self,
            objectName="action-clear-recent",
            triggered=self.clearRecent
        )

        self.__action_browse_recent = QAction(
            self.tr("Browse Recent"), self,
            objectName="action-browse-recent",
            triggered=self.browseRecent,
            shortcut=QKeySequence(
                Qt.ControlModifier | (Qt.ShiftModifier | Qt.Key_R)),
        )

        self.__recent_menu = QMenu()
        self.__recent_menu.addAction(self.__action_browse_recent)
        self.__recent_menu.addSeparator()
        self.__recent_menu.addAction(self.__action_clear_recent)

        self.__action_recent = QAction(
            self.tr("Recent"), self, objectName="action-recent"
        )
        self.__action_recent.setMenu(self.__recent_menu)

        self.__action_reload_last = QAction(
            self.tr("Reload Last"), self,
            objectName="action-reload-last",
            triggered=self.reloadLast,
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_R),
        )

        #: `Window` menu (in a OSX unified menu bar).
        self.__action_window = QAction(
            self.tr("Window"), self,
            objectName="action-window",
        )

    def setMaxRecentCount(self, count):
        """
        Set the maximum number of recent documents to keep track of.
        """
        if self.__maxRecentCount != count:
            self.__maxRecentCount = count
            del self.__recent_list[count:]

    def maxRecentCount(self):
        """
        Return the maximum number of recent documents.
        """
        return self.__maxRecentCount

    # Action getters
    # ?? def action(ActionType) -> QAction
    # ActionType = .New, .Open, .Recent, BrowseRecent, ...
    def actionNew(self):
        """Return the default 'New' document action.
        """
        return self.__action_new

    def actionOpen(self):
        """Return the default 'Open' document action.
        """
        return self.__action_open

    def actionRecent(self):
        """
        Return an QAction (with a QMenu) of recent documents.
        """
        return self.__recent

    def actionClearRecent(self):
        """
        Return the 'Clear Recent' QAction.
        """
        return self.__clear_recent

    def actionBrowseRecent(self):
        """
        Return the 'Browse Recent' QAction.
        """
        return self.__browse_recent

    def actionReloadLast(self):
        """
        Return the 'Reload Last' QAction.
        """
        return self.__reload_last

    def actionWindow(self):
        # OSX style 'Window' menu bar action.
        return self.__window

    def reloadLast(self):
        """
        Reload the last saved document.
        """
        recent = self.recentItems()
        recent = sorted(recent, key=lambda item: item.time)
        if recent:
            url = recent[-1].url
            self.openDocument(url)
            # What it should look like
            # self.openDocument(url, recent[-1].loadspec)

    def documentTypes(self):
        # type: () -> List[Type[Document]]
        """
        Return a list of all supported document types.

        Returns
        -------
        doctypes: List[Document.Type]
        """
        return list(self.__documentTypes)

    def setDocumentTypes(self, types):
        # type: (List[Type[Document]]) -> None
        self.__documentTypes = list(types)

    def setDefaultDocumentType(self, type):
        # type: (Type[Document]) -> None
        self.__defaultDocumentClass = type

    def newDocument(self, doctype=None):
        """
        Parameters
        ----------
        doctype : Optional[Document.Type]

        Returns
        -------
        doc : Document
        """
        if doctype is None:
            doctype = self.defaultDocumentClass()
        doc = doctype(self)
        self.addDocument(doc,)
        return doc

    def open(self):
        """
        The default slot for handling 'Open...' action.

        Query the user for a filename and create/load a new document from it.
        """
        dialog = self.openFileDialog()

        def whendone(path, doctype, ):
            if path:
                doc = doctype(self)
                doc.read(path)
                doc.setPath(path)
                self.addDocument(doc)

        dialog.accepted.connect(whendone)
        dialog.show()

    def openFileDialog(self, ):
        settingsqname = "{}.{}".format(__name__, type(self).__qualname__)
        dialog = QFileDialog(
            fileMode=QFileDialog.ExistingFile,
            acceptMode=QFileDialog.AcceptOpen,

        )

        s = QSettings()
        s.beginGroup(settingsqname)
        startpath = s.value("last-file-dialog-path", defaultValue="", type=str)
        selectedfilter = s.value("last-file-dialog-filter", defaultValue="", type=str)

        if startpath:
            dialog.setDirectory(startpath)

        doctypes = self.documentTypes()
        try:
            dialog.setMimeTypeFilters([doc.mimetype for doc in doctypes])
            dialog.selectMimeTypeFilter(doctypes[0].mimetype)
        except AttributeError:
            specs = types_to_filters(doctypes)
            dialog.setNameFilters(specs)
            dialog.selectNameFilter(specs[0])

        @dialog.accepted.connect
        def _():
            files = dialog.selectedFiles()
            mimetype = dialog.selectedMimeTypeFilter()
            s = QSettings()
            s.beginGroup(settingsqname)
            if files and files[0]:
                s.setValue("last-file-dialog-path", files[0])
            if mimetype:
                s.setValue("last-file-dialog-path", mimetype)

        return dialog

    def defaultDocumentClass(self):
        if self.__defaultDocumentClass:
            return self.__defaultDocumentClass
        else:
            return None

    def documentClassForUrl(self, url):
        # type: (QUrl) -> Type[Document]
        mimedb = QMimeDatabase()
        mtype = mimedb.mimeTypeForUrl(url)
        return self.documentClassForMimeType(mtype)

    def documentClassForMimeType(self, mimeType):
        # type: (QMimeType) -> Type[Document]
        for doctype in self.documentTypes():
            for mime in doctype.supportedMimeTypes():
                if mime.lower() == mimeType:
                    return doctype
        for doctype in self.documentTypes():
            for mime in doctype.supportedMimeTypes():
                if mimeType.inherits(mime):
                    return doctype

    def openDocument(self, url):
        """
        Open a new document for `url`
        """
        docclass = self.documentClassForUrl(url)
        if docclass is None:
            return None

        doc = docclass(self)

        if doc.read(url,):
            self.addDocument(doc)
            return doc
        else:
            return None

    def addDocument(self, document):
        """
        Add a document instance to this controller.
        """
        if document in self.__documents:
            raise ValueError("Document was already added.")

        self.__documents.append(document)
        document._setDocumentController(self)
        if document.path():
            self.noteRecent(document)

        self.documentOpened.emit(document)

    def closeAll(self):
        """
        Close all open documents.

        Return True if all documents accepted the close, otherwise
        return False.
        """
        for doc in self.__documents:
            if not doc.close():
                return False
        else:
            QApplication.instance().closeAllWindows()
            return True

    def currentDocument(self):
        """
        Return the current (top most) document.
        """
        window = QApplication.activeWindow()
        for doc in self.documents():
            # TODO: Search up to parent window (dialog) chain.
            if doc.widget() is window:
                return doc
        else:
            return None

    def documents(self):
        """
        Return a list of all documents.
        """
        return list(self.__documents)

    def hasModifiedDocuments(self):
        """
        Return True if any document is in a modified state.
        """
        return any(doc.isModified() for doc in self.__documents)

    def recentItems(self):
        """
        Return a list of recently open items.
        """
        return list(self.__recent)

    def noteRecent(self, item):
        # type: (Document) -> None
        """
        Add an item to the recent items list.

        Parameters
        ----------
        item : Document
        """
        if not item.url():
            return
        url = item.url()
        filename = url.fileName()
        if not item.title():
            display_name = filename
        else:
            display_name = item.title()
        path = os.path.realpath(url.toLocalFile())
        path = os.path.abspath(path)
        path = os.path.normpath(path)

        # find an item with the same path if it exists
        existing = fn.find(
            lambda ac: ac.data(Qt.UserRole) == url,
            self.__recent_group.actions()
        )
        if existing is not None:
            action = existing.val  # type: QAction
            self.__recent.pop(action.data())
            # remove from group for later re-insertion
            self.__recent_group.removeAction(action)
            self.__recent_menu.removeAction(action)
        else:
            iconprovider = QFileIconProvider()
            icon = iconprovider.icon(QFileInfo(path))
            action = QAction(
                display_name, self, icon=icon, toolTip=url.toString(),
            )
            action.triggered.connect(lambda: self.openDocument(path))

        action.setData(url)
        actions = filter(
            lambda a: isinstance(a.data(), QUrl),
            self.__recent_menu.actions()
        )

        begin = fn.find(lambda pair: pair[0] is self.__recent_begin,
                        fn.pairs(actions))
        if begin is not None:
            _, first = begin.val
        else:
            first = None

        self.__recent_menu.insertAction(first, action)
        self.__recent_group.addAction(action)
        # self.__recent.insert(0, item)

    def clearRecent(self):
        """
        Clear the list of recently opened items.
        """
        actions = self.__recent_menu.actions()
        actions = [action for action in actions
                   if isinstance(action.data(), QUrl)]

        for action in actions:
            self.__recent_menu.removeAction(action)

    def browseRecent(self):
        """
        Open dialog with recently opened items.
        """
        raise NotImplementedError


def types_to_filters(types):
    # type: (List[Document.Type]) -> List[str]
    return ["{} (*.{})".format(t.name, t.extension)
            for t in types]


import operator
import itertools

from functools import partial


class fn(object):
    Some = namedtuple("Some", ["val"])

    @staticmethod
    def index_in(el, sequence):
        return fn.index(partial(operator.eq, el), sequence)

    @staticmethod
    def index(pred, sequence):
        try:
            return fn.Some(next(i for i, v in enumerate(sequence) if pred(v)))
        except StopIteration:
            return None

    @staticmethod
    def find(pred, sequence):
        try:
            return fn.Some(next(v for v in sequence if pred(v)))
        except StopIteration:
            return None

    @staticmethod
    def pairs(sequence):
        s1, s2 = itertools.tee(sequence)
        try:
            next(s2)
        except StopIteration:
            return zip((), ())
        else:
            return zip(s1, s2)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    app = QApplication(list(argv))
    argv = app.arguments()

    controller = DocumentController()
    controller.setDefaultDocumentType(TextDocument)
    controller.setDocumentTypes([SvgDocument, ImageDocument, TextDocument,])
    if len(argv) > 1:
        fname = argv[1]
        doc = controller.openDocument(QUrl.fromLocalFile(fname))
    else:
        doc = controller.newDocument()
    if doc is not None:
        view = doc.widget()
        view.show()
        return app.exec_()
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
