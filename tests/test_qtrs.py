import unittest
from secScraper import qtrs
import os


class TestQtrs(unittest.TestCase):
    def setUp(self):
        self.s = dict()
        self.s['list_qtr'] = [(2010, k) for k in range(1, 5)] + [(2011, k) for k in range(1, 5)]

    def test_create_qtr_list_same_year(self):
        test = qtrs.create_qtr_list([(2018, 1), (2018, 4)])
        self.assertEqual(test, [(2018, 1), (2018, 2), (2018, 3), (2018, 4)])

    def test_create_qtr_list_across_years(self):
        test = qtrs.create_qtr_list([(2016, 2), (2017, 3)])
        self.assertEqual(test, [(2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3)])

    def test_yearly_qtr_list_same_year(self):
        test = qtrs.yearly_qtr_list([(2016, 2), (2016, 2)])
        self.assertEqual(test, [(2016, 2)])

    def test_yearly_qtr_list_across_years(self):
        test = qtrs.yearly_qtr_list([(2015, 2), (2016, 3)])
        self.assertEqual(test, [[(2015, 2), (2015, 3), (2015, 4)], [(2016, 1), (2016, 2), (2016, 3)]])

    def test_is_downloaded_true(self):
        home = os.path.expanduser('~')
        path_temp_file = os.path.join(home, "temp_test_is_downloaded.temp")
        with open(path_temp_file, 'w'):
            pass
        test = qtrs.is_downloaded(path_temp_file)
        os.remove(path_temp_file)
        self.assertTrue(test)

    def test_is_downloaded_false(self):
        home = os.path.expanduser('~')
        test = qtrs.is_downloaded(os.path.join(home, "ahsbxaksjhbxhjx.txt"))
        self.assertFalse(test)  # That file is unlikely to exist

    def test_previous_qtr_same_year(self):
        test = qtrs.previous_qtr((2011, 1), self.s)
        self.assertEqual(test, (2010, 4))

    def test_previous_qtr_across_years(self):
        test = qtrs.previous_qtr((2011, 4), self.s)
        self.assertEqual(test, (2011, 3))

    def test_qtr_to_day_first(self):
        test = qtrs.qtr_to_day((2008, 1), 'first', date_format='string')
        self.assertEqual(test, '20080101')

    def test_qtr_to_day_last(self):
        test = qtrs.qtr_to_day((2005, 4), 'last', date_format='string')
        self.assertEqual(test, '20051231')


if __name__ == '__main__':
    unittest.main()
