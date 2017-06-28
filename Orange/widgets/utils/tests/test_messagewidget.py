from AnyQt.QtTest import QTest

from Orange.widgets.tests.base import GuiTest

from Orange.widgets.utils.messagewidget import MessagesWidget, Message


class TestMessageWidget(GuiTest):
    def test_widget(self):
        w = MessagesWidget()
        w.setMessage(0, Message())
        self.assertTrue(w.summarize().isEmpty())
        self.assertSequenceEqual(w.messages(), [Message()])
        w.setMessage(0, Message(Message.Warning, text="a"))
        self.assertFalse(w.summarize().isEmpty())
        self.assertEqual(w.summarize().severity, Message.Warning)
        self.assertEqual(w.summarize().text, "a")
        w.setMessage(1, Message(Message.Error, text="#error#"))
        self.assertEqual(w.summarize().severity, Message.Error)
        self.assertTrue(w.summarize().text.startswith("#error#"))
        self.assertSequenceEqual(
            w.messages(),
            [Message(Message.Warning, text="a"),
             Message(Message.Error, text="#error#")])
        w.clear()
        self.assertEqual(len(w.messages()), 0)
        self.assertTrue(w.summarize().isEmpty())
