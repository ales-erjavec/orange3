import os
import sys
import enum
import base64
from itertools import chain
from operator import attrgetter
from xml.sax.saxutils import escape
from collections import OrderedDict
from typing import (
    NamedTuple, Tuple, List, Dict, Iterable, Union, Optional, Hashable
)

from AnyQt.QtCore import Qt, QSize, QBuffer
from AnyQt.QtGui import (
    QIcon, QPixmap, QPainter, QPalette, QLinearGradient, QBrush
)
from AnyQt.QtWidgets import (
    QWidget, QLabel, QSizePolicy, QStyle, QHBoxLayout, QMessageBox,
    QMenu, QWidgetAction, QStyleOption, QApplication
)
from AnyQt.QtCore import pyqtSignal as Signal


def image_data(pm):
    # type: (QPixmap) -> str
    """
    Render the contents of the pixmap as a data URL (RFC-2397)

    Parameters
    ----------
    pm : QPixmap

    Returns
    -------
    datauri : str
    """
    pm = QPixmap(pm)
    device = QBuffer()
    assert device.open(QBuffer.ReadWrite)
    pm.save(device, b'png')
    device.close()
    data = bytes(device.data())
    payload = base64.b64encode(data).decode("ascii")
    return "data:image/png;base64," + payload


class Severity(enum.IntEnum):
    Information = QMessageBox.Information
    Warning = QMessageBox.Warning
    Error = QMessageBox.Critical


# Maybe a better way: `summary` (brief - inline description), `detailed`
# (extra text to append)
class Message(
    NamedTuple(
        "Message", [
            ("severity", Severity),
            ("icon", QIcon),
            ("text", str),
            ("informativeText", str),
            ("detailedText", str),
            ("textFormat", Qt.TextFormat)
        ])):
    """
    A stateful message/notification.

    Parameters
    ----------
    severity : Message.Severity
        Severity level (default: Information).
    icon : QIcon
        Associated icon. If empty the QStyle.standardIcon() will be used based
        on severity.
    text : str
        Short message text.
    informativeText : str
        Extra informative text to append to `text` (space permitting).
    detailedText : str
        Extra detailed text (e.g. exception traceback)
    textFormat : Qt.TextFormat
        If `Qt.RichText` then the contents of `text`, `informativeText` and
        `detailedText` will be rendered as html instead of plain text.

    """
    Severity = Severity
    Warning = Severity.Warning
    Information = Severity.Information
    Error = Severity.Error

    def __new__(cls, severity=Severity.Information, icon=QIcon(), text="",
                informativeText="", detailedText="", textFormat=Qt.PlainText):
        return super().__new__(cls, Severity(severity), icon, text,
                               informativeText, detailedText, textFormat)

    def asHtml(self):
        # type: () -> str
        """
        Render the message as an HTML fragment.
        """
        if self.textFormat == Qt.RichText:
            render = lambda t: t
        else:
            render = lambda t: ('<span style="white-space: pre">{}</span>'
                                .format(escape(t)))

        def iconsrc(message):
            # type: (Message) -> str
            """
            Return an image src url for message icon or None if not available.
            """
            icon = message_icon(message)
            pm = icon.pixmap(12, 12)
            return image_data(pm)

        parts = [
            '<div class="message {}">'.format(self.severity.name.lower()),
            # '<img src="{}"/>'.format(iconsrc(self)),
            '<nobr class="field-text">',
            '{}'.format(render(self.text)),
            '</nobr>',
        ]
        if self.informativeText:
            parts += ['<br/><span class="field-informative-text">{}</span>'
                      .format(render(self.informativeText))]
        if self.detailedText:
            parts += ['<blockquote class="field-detailed-text">{}</blockquote>'
                      .format(render(self.detailedText))]
        parts += ['</div>']
        return "\n".join(parts)

    def isEmpty(self):
        return not self.text and self.icon.isNull() and \
               not self.informativeText and not self.detailedText


def standard_pixmap(severity):
    maping = {
        Severity.Information: QStyle.SP_MessageBoxInformation,
        Severity.Warning: QStyle.SP_MessageBoxWarning,
        Severity.Error: QStyle.SP_MessageBoxCritical,
    }
    return maping[severity]


def message_icon(message, style=None):
    # type: (Message, Optional[QStyle]) -> QIcon
    """
    Return the resolved icon for the message.

    Parameters
    ----------
    message
    style

    Returns
    -------

    """
    if style is None and QApplication.instance() is not None:
        style = QApplication.style()
    if message.icon.isNull():
        icon = style.standardIcon(standard_pixmap(message.severity))
    else:
        icon = message.icon
    return icon


def summarize(messages):
    # type: (List[Message]) -> Message
    """
    Summarize a list of messages into a single message instance

    Parameters
    ----------
    messages

    Returns
    -------

    """
    if not messages:
        raise ValueError("Empty messages list")

    if len(messages) == 1:
        return messages[0]

    def categorize(messages):
        # type: (List[Message]) -> Tuple[Message, List[Message], List[Message], List[Message]]
        errors = [m for m in messages if m.severity == Severity.Error]
        warnings = [m for m in messages if m.severity == Severity.Warning]
        info = [m for m in messages if m.severity == Severity.Information]
        lead = None
        if len(errors) == 1:
            lead = errors.pop(-1)
        elif not errors and len(warnings) == 1:
            lead = warnings.pop(-1)
        elif not errors and not warnings and len(info) == 1:
            lead = info.pop(-1)
        return lead, errors, warnings, info

    lead, errors, warnings, info = categorize(messages)
    severity = Severity.Information
    icon = QIcon()
    leading_text = ""
    text_parts = []
    if lead is not None:
        severity = lead.severity
        icon = lead.icon
        leading_text = lead.text
    else:
        if errors:
            severity = Severity.Error
        elif warnings:
            severity = Severity.Warning

    def format_plural(fstr, items, *args, **kwargs):
        return fstr.format(len(items), *args,
                           s="s" if len(items) != 1 else "",
                           **kwargs)
    if errors:
        text_parts.append(format_plural("{} error{s}", errors))
    if warnings:
        text_parts.append(format_plural("{} warning{s}", warnings))
    if info:
        if not (errors and warnings and lead):
            text_parts.append(format_plural("{} message{s}", info))
        else:
            text_parts.append(format_plural("{} other", info))

    if leading_text:
        text = leading_text
        if text_parts:
            text = text + " (" + ", ".join(text_parts) + ")"
    else:
        text = ", ".join(text_parts)
    detailed = "<hr/>".join(m.asHtml()
                            for m in chain([lead], errors, warnings, info)
                            if m is not None and not m.isEmpty())
    return Message(severity, icon, text, detailedText=detailed,
                   textFormat=Qt.RichText)


def _resource_path(path):
    return os.path.join(os.path.dirname(__file__), path)


class MessagesWidget(QWidget):
    linkActivated = Signal(str)
    linkHovered = Signal(str)

    Severity = Severity
    Message = Message

    def __init__(self, parent=None, openExternalLinks=False, **kwargs):
        kwargs.setdefault(
            "sizePolicy",
            QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        )
        super().__init__(parent, **kwargs)
        self.__openExternalLinks = openExternalLinks
        self.__messages = OrderedDict()  # type: Dict[Hashable, Message]
        #: The full (joined all messages text - rendered as html)
        self.__fulltext = ""  # type: str
        self.__iconlabel = QLabel(
            sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed),
        )
        self.__textlabel = QLabel(
            wordWrap=False,
            textInteractionFlags=Qt.LinksAccessibleByMouse,
            openExternalLinks=self.__openExternalLinks,
            sizePolicy=QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        )
        icon = QIcon(_resource_path("../icons/popup-indicator-details.svg"))
        self.__popupicon = QLabel(
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum),
            pixmap=icon.pixmap(15, 15),
            visible=False,
        )
        # self.__popupicon.setText(
        #     '<span style="font-weight: 900">\N{VERTICAL ELLIPSIS}</b>')
        self.__textlabel.linkActivated.connect(self.linkActivated)
        self.__textlabel.linkHovered.connect(self.linkHovered)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(2, 0, 2, 0)
        self.layout().setSpacing(0)
        self.layout().addWidget(self.__iconlabel)
        self.layout().addSpacing(4)
        self.layout().addWidget(self.__textlabel)
        self.layout().addWidget(self.__popupicon)
        self.__textlabel.setAttribute(Qt.WA_MacSmallSize)
        self.__popupicon.setAttribute(Qt.WA_MacSmallSize)

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(0, 20))

    def setMessages(self, messages):
        # type: (Union[Iterable[Tuple[Hashable, Message]], Dict[Hashable, Message]]) -> None
        messages = OrderedDict(messages)
        # self.__messages.clear()
        self.__messages.update(messages)
        self.__update()

    def setMessage(self, message_id, message):
        # type: (Hashable, Message) -> None
        """
        Add a `message` for `message_id` to the current display.
        """
        self.__messages[message_id] = message
        self.__update()

    def removeMessage(self, message_id):
        # type: (Hashable) -> None
        """
        Remove message from the display.
        """
        del self.__messages[message_id]
        self.__update()

    def clear(self):
        # type: () -> None
        """
        Clear all messages.
        """
        self.__messages.clear()
        self.__update()

    def messages(self):
        # type: () -> List[Message]
        return list(self.__messages.values())

    def summarize(self):
        # type: () -> Message
        """
        Summarize all the messages into a single message.
        """
        messages = [m for m in self.__messages.values() if not m.isEmpty()]
        if messages:
            return summarize(messages)
        else:
            return Message()

    def __update(self):
        """
        Update the current display state.
        """
        self.ensurePolished()
        style = self.style()

        def styleicon(severity):
            maping = {
                Severity.Information: QStyle.SP_MessageBoxInformation,
                Severity.Warning: QStyle.SP_MessageBoxWarning,
                Severity.Error: QStyle.SP_MessageBoxCritical,
            }
            return maping[severity]
        summary = self.summarize()
        # icon = QIcon()
        icon = message_icon(summary)
        # if not summary.isEmpty():
        #     if summary.icon.isNull():
        #         icon = self.style().standardIcon(styleicon(summary.severity))
        #     else:
        #         icon = summary.icon

        if not icon.isNull():
            size = style.pixelMetric(QStyle.PM_SmallIconSize)
            pm = icon.pixmap(size, size)
        else:
            pm = QPixmap()
        self.__iconlabel.setPixmap(pm)
        self.__iconlabel.setVisible(not pm.isNull())
        self.__textlabel.setTextFormat(summary.textFormat)
        self.__textlabel.setText(summary.text)
        messages = [m for m in self.__messages.values() if not m.isEmpty()]
        if messages:
            messages = sorted(messages,
                              key=attrgetter("severity"),
                              reverse=True)
            fulltext = "<hr/>".join(m.asHtml() for m in messages)
        else:
            fulltext = ""
        self.__fulltext = fulltext
        self.setToolTip(fulltext)
        self.__popupicon.setVisible(bool(messages))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.__fulltext:
                popup = QMenu(self)
                l = QLabel(
                    self, textInteractionFlags=Qt.TextBrowserInteraction,
                    openExternalLinks=self.__openExternalLinks,
                    text=self.__fulltext
                )
                l.linkActivated.connect(self.linkActivated)
                l.linkHovered.connect(self.linkHovered)
                action = QWidgetAction(popup)
                action.setDefaultWidget(l)
                popup.addAction(action)
                popup.popup(event.globalPos(), action)
                event.accept()
            return
        else:
            super().mousePressEvent(event)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.update()

    def changeEvent(self, event):
        super().changeEvent(event)
        self.update()

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        if self.__fulltext and \
                opt.state & QStyle.State_MouseOver and \
                opt.state & QStyle.State_Active:
            g = QLinearGradient()
            g.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
            base = opt.palette.color(QPalette.Window)
            g.setColorAt(0, base.lighter(200))
            g.setColorAt(0.6, base)
            g.setColorAt(1.0, base.lighter(200))

            p = QPainter(self)
            p.setBrush(QBrush(g))
            p.setOpacity(0.2)
            p.drawRect(opt.rect)


def main(argv=None):
    from AnyQt.QtWidgets import QVBoxLayout, QCheckBox, QStatusBar
    app = QApplication(list(argv) if argv else [])
    l1 = QVBoxLayout()
    l1.setContentsMargins(0, 0, 0, 0)
    blayout = QVBoxLayout()
    l1.addLayout(blayout)
    sb = QStatusBar()

    w = QWidget()
    w.setLayout(l1)
    messages = [
        Message(Severity.Error, text="Encountered a HCF",
                detailedText="<em>AAA! It burns.</em>",
                textFormat=Qt.RichText),
        Message(Severity.Warning,
                text="ACHTUNG!",
                detailedText=
                    "<div style=\"color: red\">DAS KOMPUTERMASCHINE IST "
                   "NICHT FÜR DER GEFINGERPOKEN</div>",
                textFormat=Qt.RichText),
        Message(Severity.Information,
                text="The rain in spain falls mostly on the plain",
                informativeText="<a href=\"https://www.google.si/search?q=Average+Yearly+Precipitation+in+Spain\">Link</a>",
                textFormat=Qt.RichText),
        Message(Severity.Error,
                text="I did not do this!",
                informativeText="The computer made suggestions...",
                detailedText="... and the default options was yes."),
        Message(),
    ]
    mw = MessagesWidget(openExternalLinks=True)
    for i, m in enumerate(messages):
        cb = QCheckBox(m.text)

        def toogled(state, i=i, m=m):
            if state:
                mw.setMessage(i, m)
            else:
                mw.removeMessage(i)
        cb.toggled[bool].connect(toogled)
        blayout.addWidget(cb)

    sb.addWidget(mw)
    w.layout().addWidget(sb, 0)
    w.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main(sys.argv))
