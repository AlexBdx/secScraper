import unittest
from secScraper import parser


class TestParser(unittest.TestCase):

    def setUp(self):
        self.first_markers_dirty = {
            '_i_1': [(5241, 5259)],
            '_i_2': [(32578, 32597)],
            '_i_3': [(68076, 68097)],
            '_i_4': [(69489, 69506)],
            'ii_1': [(70893, 70907)],
            'ii_1a': [(4963, 4978), (54617, 54632), (71237, 71251)],
            'ii_2': [(71464, 71485)],
            'ii_3': [(71541, 71558)],
            'ii_5': [(71623, 71637)],
            'ii_6': [(71685, 71702)]
        }
        self.first_markers_clean = {
            '_i_1': [(5241, 5259)],
            '_i_2': [(32578, 32597)],
            '_i_3': [(68076, 68097)],
            '_i_4': [(69489, 69506)],
            'ii_1': [(70893, 70907)],
            'ii_1a': [(71237, 71251)],
            'ii_2': [(71464, 71485)],
            'ii_3': [(71541, 71558)],
            'ii_5': [(71623, 71637)],
            'ii_6': [(71685, 71702)]
        }

    def test_clean_first_markers(self):
        test = parser.clean_first_markers(self.first_markers_dirty)
        self.assertEqual(test, self.first_markers_clean)


if __name__ == '__main__':
    unittest.main()
