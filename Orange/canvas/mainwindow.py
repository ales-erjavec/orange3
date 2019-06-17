from orangewidget.workflow.mainwindow import OWCanvasMainWindow
from Orange.canvas.utils.overlay import NotificationOverlay


class MainWindow(OWCanvasMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notification_overlay = NotificationOverlay(self.scheme_widget)

    def closeEvent(self, event):
        self.notification_overlay.close()
        super().closeEvent(event)
