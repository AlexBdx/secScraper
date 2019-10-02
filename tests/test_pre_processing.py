import unittest
from secScraper import pre_processing


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.s = {'list_qtr': [
            (2010, 1),
            (2010, 2),
            (2010, 3),
            (2010, 4),
            (2011, 1),
            (2011, 2),
            (2011, 3),
            (2011, 4),
            (2012, 1),
            (2012, 2),
            (2012, 3),
            (2012, 4)
        ]}
        self.qs1 = {
            (2010, 1): [{'type': '10-K', 'published': (2010, 2, 26), 'qtr': (2010, 1)}],
            (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}],
            (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}],
            (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}],
            (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}],
            (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}],
            (2011, 3): [{'type': '10-Q', 'published': (2011, 8, 5), 'qtr': (2011, 3)}],
            (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}],
            (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}],
            (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}],
            (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}],
            (2012, 4): [{'type': '10-Q', 'published': (2012, 11, 2), 'qtr': (2012, 4)}]
        }
        self.qs2 = {
            (2010, 1): [],
            (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}],
            (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}],
            (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}],
            (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}],
            (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}],
            (2011, 3): [{'type': '10-Q', 'published': (2011, 8, 5), 'qtr': (2011, 3)}],
            (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}],
            (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}],
            (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}],
            (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}],
            (2012, 4): [{'type': '10-Q', 'published': (2012, 11, 2), 'qtr': (2012, 4)}]
        }
        self.qs3 = {
            (2010, 1): [{'type': '10-K', 'published': (2010, 2, 26), 'qtr': (2010, 1)}],
            (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}],
            (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}],
            (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}],
            (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}],
            (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}],
            (2011, 3): [{'type': '10-Q', 'published': (2011, 8, 5), 'qtr': (2011, 3)}],
            (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}],
            (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}],
            (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}],
            (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}],
            (2012, 4): []
        }
        self.qs4 = {
            (2010, 1): [{'type': '10-K', 'published': (2010, 2, 26), 'qtr': (2010, 1)}],
            (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}],
            (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}],
            (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}],
            (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}],
            (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}],
            (2011, 3): [],
            (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}],
            (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}],
            (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}],
            (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}],
            (2012, 4): [{'type': '10-Q', 'published': (2012, 11, 2), 'qtr': (2012, 4)}]
        }
        self.qs5 = {
            (2010, 1): [{'type': '10-K', 'published': (2010, 2, 26), 'qtr': (2010, 1)}],
            (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}],
            (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}],
            (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}],
            (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}],
            (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)},
                        {'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}],
            (2011, 3): [{'type': '10-Q', 'published': (2011, 8, 5), 'qtr': (2011, 3)}],
            (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}],
            (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}],
            (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}],
            (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}],
            (2012, 4): [{'type': '10-Q', 'published': (2012, 11, 2), 'qtr': (2012, 4)}]
        }
        self.qs6 = {
            (2010, 1): [],
            (2010, 2): [],
            (2010, 3): [],
            (2010, 4): [],
            (2011, 1): [],
            (2011, 2): [],
            (2011, 3): [],
            (2011, 4): [],
            (2012, 1): [],
            (2012, 2): [],
            (2012, 3): [],
            (2012, 4): []
        }
        self.list_qs = [
            (self.qs1, True),
            (self.qs2, True),
            (self.qs3, True),
            (self.qs4, False),
            (self.qs5, False),
            (self.qs6, False)
        ]

    def test_check_report_continuity_all_qtr_covered(self):
        test = pre_processing.check_report_continuity(self.list_qs[0][0], self.s)
        self.assertTrue(test, self.list_qs[0][1])

    def test_check_report_continuity_not_listed_at_start(self):
        test = pre_processing.check_report_continuity(self.list_qs[1][0], self.s)
        self.assertTrue(test, self.list_qs[1][1])

    def test_check_report_continuity_delisted_before_end(self):
        test = pre_processing.check_report_continuity(self.list_qs[2][0], self.s)
        self.assertTrue(test, self.list_qs[2][1])

    def test_check_report_continuity_missing_one_report(self):
        test = pre_processing.check_report_continuity(self.list_qs[3][0], self.s)
        self.assertFalse(test, self.list_qs[3][1])

    def test_check_report_continuity_multiple_reports_in_a_qtr(self):
        test = pre_processing.check_report_continuity(self.list_qs[4][0], self.s)
        self.assertFalse(test, self.list_qs[4][1])

    def test_check_report_continuity_no_reports_at_all(self):
        test = pre_processing.check_report_continuity(self.list_qs[5][0], self.s)
        self.assertFalse(test, self.list_qs[5][1])


if __name__ == '__main__':
    unittest.main()
