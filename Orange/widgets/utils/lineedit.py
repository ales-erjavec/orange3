from AnyQt.QtCore import Slot, Signal, Property
from AnyQt.QtGui import QValidator
from AnyQt.QtWidgets import QLineEdit


CSS = """
QLineEdit.invalid-input {
    color: black;
    background: #ed1515;
}

QLineEdit.intermediate-input:focus {
    color: black;
    background: #fffc4b;
}

QLineEdit.intermediate-input {
    color: black;
    background: #ed1515;
}
"""


class LineEdit(QLineEdit):
    """
    A QLineEdit with visual feedback for validator state changes.
    """
    #: Signal emitted when the QValidator state changes
    validationStateChanged = Signal(QValidator.State)

    def __init__(self, *args, **kwargs):
        self.__validationState = QValidator.Acceptable
        self.__modified = False
        self.__currText = ""
        text = kwargs.pop("text", None)
        # These signals must not be connected before the __init__
        textEdited = kwargs.pop("textEdited", None)
        textChanged = kwargs.pop("textChanged", None)
        super().__init__(*args, **kwargs)
        self.setStyleSheet(CSS)
        # These connections must be connected and fire before client supplied
        # ones (possible kwargs).
        self.textChanged.connect(self.__on_textChanged)
        self.textEdited.connect(self.__on_textEdited)
        if textEdited is not None:
            self.textEdited.connect(self.__on_textEdited)
        if textChanged is not None:
            self.textChanged.connect(self.__on_textChanged)
        if text is not None:
            self.setText(text)

    @Slot()
    def __on_textEdited(self):
        self.setModified(True)

    @Slot()
    def __on_textChanged(self):
        if not self.__in_setText:
            self.setModified(True)
        self.__updateValidationState()

    def __updateValidationState(self):
        text = self.text()
        validator = self.validator()
        if validator is not None:
            state, _, _ = validator.validate(text, 0)
        else:
            state = QValidator.Acceptable
        if state == QValidator.Invalid:
            self.setProperty("class", "invalid-input")
        if state == QValidator.Intermediate:
            self.setProperty("class", "intermediate-input")
        else:
            self.setProperty("class", "")

        s = self.style()
        s.unpolish(self)
        s.polish(self)

        if state != self.__validationState:
            self.__validationState = state
            self.validationStateChanged.emit(state)

    def setValidator(self, validator: QValidator) -> None:
        super().setValidator(validator)
        self.__updateValidationState()

    def validationState(self) -> QValidator.State:
        return self.__validationState

    __in_setText = False

    # setText, isModified, ... are reimplemented due to QTBUG-49295
    def setText(self, text: str) -> None:
        old = self.__in_setText
        self.__in_setText = True
        self.__currText = text
        super().setText(text)
        self.__in_setText = old
        self.setModified(False)

    def isModified(self):
        return self.__modified

    def setModified(self, modified: bool) -> None:
        super().setModified(modified)
        if self.__modified != modified:
            self.__modified = modified
            self.modifiedChanged.emit(modified)

    modifiedChanged = Signal(bool)
    modified_ = Property(bool, isModified, setModified, notify=modifiedChanged)
