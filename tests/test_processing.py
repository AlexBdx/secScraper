import unittest
from secScraper import processing


class TestProcessing(unittest.TestCase):

    def test_normalize_texts(self):
        current_text = 'Hello       Sir.'
        previous_text = "How are you?\r\n I'm good, \t and you?"
        test = processing.normalize_texts(current_text, previous_text)
        self.assertEqual(test, ('Hello Sir.', "How are you? I'm good, and you?"))


if __name__ == '__main__':
    unittest.main()
