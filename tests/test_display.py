import unittest
from secScraper import display


class TestDisplay(unittest.TestCase):

    def test_run_from_ipython(self):
        try:
            __IPYTHON__
            self.assertTrue(display.run_from_ipython())
        except NameError:
            self.assertFalse(display.run_from_ipython())


if __name__ == '__main__':
    unittest.main()
