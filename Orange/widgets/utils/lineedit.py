from AnyQt.QtCore import Slot, Signal
from AnyQt.QtGui import QValidator
from AnyQt.QtWidgets import QLineEdit


CSS = """
QLineEdit.invalid-input {
    color: black;
    background: red;
}

QLineEdit.intermediate-input:focus {
    color: black;
    background: yellow;
}

QLineEdit.intermediate-input {
    color: black;
    background: red;
}
"""


class LineEdit(QLineEdit):
    #: Signal emitted when the QValidator state changes
    validationStateChanged = Signal(QValidator.State)

    def __init__(self, *args, **kwargs):
        self.__validationState = QValidator.Acceptable
        text = kwargs.pop("text", None)
        super().__init__(*args, **kwargs)
        self.setStyleSheet(CSS)
        self.textChanged.connect(self.__on_textChanged)
        if text is not None:
            self.setText(text)

    @Slot(str)
    def __on_textChanged(self, text: str):
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

    def validationState(self) -> QValidator.State:
        return self.__validationState
