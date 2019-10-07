import unittest
from secScraper import metrics, processing


class TestMetrics(unittest.TestCase):

    def test_diff_jaccard_high(self):
        """
        Test for the diff_jaccard function.

        :return: bool
        """
        da = "We expect demand to increase."
        db = "We expect worldwide demand to increase."
        da = processing.normalize_text(da)
        db = processing.normalize_text(db)
        test = metrics.diff_jaccard(da, db)
        self.assertEqual(round(test, 2), 0.83)

    def test_diff_jaccard_low(self):
        """
        Test for the diff_jaccard function.

        :return: bool
        """
        da = "We expect demand to increase."
        dc = "We expect weakness in sales."
        da = processing.normalize_text(da)
        dc = processing.normalize_text(dc)
        test = metrics.diff_jaccard(da, dc)
        self.assertEqual(round(test, 2), 0.25)

    def test_diff_cosine_tf_high(self):
        """
        Test for the Cosine TF function.

        :return: bool
        """
        da = "We expect demand to increase."
        db = "We expect worldwide demand to increase."
        test = metrics.diff_cosine_tf(da, db)
        self.assertEqual(round(test, 2), 0.91)

    def test_diff_cosine_tf_low(self):
        """
        Test for the Cosine TF function.

        :return: bool
        """
        da = "We expect demand to increase."
        dc = "We expect weakness in sales."
        test = metrics.diff_cosine_tf(da, dc)
        self.assertEqual(round(test, 2), 0.40)

    def test_diff_minEdit_high(self):
        """
        Test for the diff_minEdit function.

        :return: bool
        """
        da = "We expect demand to increase."
        db = "We expect worldwide demand to increase."
        test = metrics.diff_minEdit(da, db)
        self.assertEqual(round(test, 2), 0.91)

    def test_diff_minEdit_low(self):
        """
        Test for the diff_minEdit function.

        :return: bool
        """
        da = "We expect demand to increase."
        dc = "We expect weakness in sales."
        test = metrics.diff_minEdit(da, dc)
        self.assertEqual(round(test, 2), 0.30)

    def test_diff_simple_high(self):
        """
        Test for the diff_simple function.

        :return: bool
        """
        da = "We expect demand to increase."
        db = "We expect worldwide demand to increase."
        test = metrics.diff_simple(da, db)
        self.assertEqual(round(test, 2), 0.85)

    def test_diff_simple_low(self):
        """
        Test for the diff_simple function.

        :return: bool
        """
        da = "We expect demand to increase."
        dc = "We expect weakness in sales."
        test = metrics.diff_simple(da, dc)
        self.assertEqual(round(test, 2), 0.67)





if __name__ == '__main__':
    unittest.main()
