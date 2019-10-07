import unittest
from secScraper import processing


class TestProcessing(unittest.TestCase):
    def setUp(self):
        self.s = {
            'straight_table': {
            '10-K': ['1', '1a', '1b', '2', '3', '4', '5', '6', '7', '7a', '8', '9', '9a', '9b', '10', '11', '12', '13',
                     '14', '15'],
            '10-Q': ['_i_1', '_i_2', '_i_3', '_i_4', 'ii_1', 'ii_1a', 'ii_2', 'ii_3', 'ii_4', 'ii_5', 'ii_6']
        },
            'metrics': ['diff_jaccard', 'sing_sentiment'],
            'epsilon': 0.0001
        }
        self.input_text = "Hello       Sir. How are you?\r\n I'm good, \t and you? That isn't clear."

    def test_normalize_text_no_options(self):  # There could be many more tests...
        test = processing.normalize_text(self.input_text)
        expected_result = ['hello','sir','.', 'how', 'are', 'you', '?', 'i', "'m", 'good', ',',
                           'and', 'you', '?', 'that', 'is', "n't", 'clear', '.']
        self.assertEqual(test, expected_result)


    def test_average_report_scores_10k(self):
        report_type = '10-K'
        sections_to_consider = self.s['straight_table'][report_type]
        nb_sections = len(sections_to_consider)
        # Create a fake result section with identical values for a given metric
        result = {section: {
            m: (idx + 1) / len(self.s['metrics']) for idx, m in enumerate(self.s['metrics'])
        } for section in sections_to_consider}
        wc = (2, 1)
        word_count = {section: wc for section in sections_to_consider}
        idx = 4  # We introduce a single perturbation here
        for m in self.s['metrics']:
            result[sections_to_consider[idx]][m] = 0
            word_count[sections_to_consider[idx]] = (wc[0] * (nb_sections - 1), wc[1] * nb_sections - 1)

        test = processing.average_report_scores(result, word_count, self.s)
        expected_result = {m: (1/2)*(idx + 1) / len(self.s['metrics']) for idx, m in enumerate(self.s['metrics'])}

        for k in test.keys():
            self.assertAlmostEqual(test[k], expected_result[k])

    def test_average_report_scores_10q(self):
        report_type = '10-Q'
        sections_to_consider = self.s['straight_table'][report_type]
        nb_sections = len(sections_to_consider)
        # Create a fake result section with identical values for a given metric
        result = {section: {
            m: (idx + 1) / len(self.s['metrics']) for idx, m in enumerate(self.s['metrics'])
        } for section in sections_to_consider}
        wc = (2, 1)
        word_count = {section: wc for section in sections_to_consider}
        idx = 4  # We introduce a single perturbation here
        for m in self.s['metrics']:
            result[sections_to_consider[idx]][m] = 0
            word_count[sections_to_consider[idx]] = (wc[0] * (nb_sections - 1), wc[1] * nb_sections - 1)

        test = processing.average_report_scores(result, word_count, self.s)
        expected_result = {m: (1/2)*(idx + 1) / len(self.s['metrics']) for idx, m in enumerate(self.s['metrics'])}

        for k in test.keys():
            self.assertAlmostEqual(test[k], expected_result[k])


if __name__ == '__main__':
    unittest.main()
