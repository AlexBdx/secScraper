import unittest
from secScraper import post_processing
from datetime import datetime


class TestPostprocessing(unittest.TestCase):
    def setUp(self):
        self.s = {
            'time_range': [(2009, 3), (2009, 4)],
            'list_qtr': [(2009, 3), (2009, 4)],
            'bin_labels': ['Q1'],
            'bin_count': 1,
            'lag': 0,
            'metrics': ['diff_jaccard', 'sing_sentiment'],
            'pf_init_value': 100
        }

        self.pf_scores = {
            'diff_jaccard':
                {
                    'Q1':
                        {
                            (2009, 3):
                                [
                                    [851968, 0.016382932139384836, 0, 0],
                                    [1048685, 0.04691367902910114, 0, 0],
                                    [884905, 0.03741568087507438, 0, 0],
                                    [721083, 0.12758043213199938, 0, 0],
                                    [10, 0.12758043213199938, 0, 0]
                                ],
                            (2009, 4):
                                [
                                    [851968, 0.016382932139384836, 0, 0],
                                    [1048685, 0.04691367902910114, 0, 0],
                                    [884905, 0.03741568087507438, 0, 0],
                                    [721083, 0.12758043213199938, 0, 0],
                                    [10, 0.12758043213199938, 0, 0]
                                ]
                        }
                }
        }
        self.lookup = {
            851968: 'A',
            1048685: 'B',
            884905: 'C',
            721083: 'D',
            0: 'E',
            10: 'F'
        }
        self.stock_data = {
            'A': {
                datetime.strptime('20090701', '%Y%m%d').date(): [2, 750],
                datetime.strptime('20091001', '%Y%m%d').date(): [1, 150]
            },
            'B': {
                datetime.strptime('20090701', '%Y%m%d').date(): [1, 150],
                datetime.strptime('20091001', '%Y%m%d').date(): [1, 750]
            },
            'C': {
                datetime.strptime('20090701', '%Y%m%d').date(): [1, 50],
                datetime.strptime('20091001', '%Y%m%d').date(): [1, 50]
            },
            'D': {
                datetime.strptime('20090701', '%Y%m%d').date(): [1, 50],
                datetime.strptime('20091001', '%Y%m%d').date(): [1, 50]
            },
            'E': {
                datetime.strptime('20090701', '%Y%m%d').date(): [1, 1000],
                datetime.strptime('20091001', '%Y%m%d').date(): [1, 1000]
            }
        }

    def test_calculate_portfolio_value_balanced(self):
        """
        This rather complex, not so unitary, test validates that the balanced portfolio really is. It asserts that:
        (money invested in a stock)/(money available to invest) = (stock's market cap)/(sum market caps invested in)
        The assertion relies on a assertAlmostEqual to the 2nd decimal place (so basically to the cent).

        :return:
        """
        # This is common between balanced and unbalanced
        tax_rate = 0
        pf_values = {m: 0 for m in self.s['metrics'][:-1]}
        for m in self.s['metrics'][:-1]:
            pf_values[m] = {q: {qtr: [0, tax_rate, 0] for qtr in self.s['list_qtr']} for q in self.s['bin_labels']}

        pf_scores = post_processing.calculate_portfolio_value(self.pf_scores, pf_values, self.lookup, self.stock_data, self.s)
        # ---------------------------------------------------

        # Now we test the balanced portfolio!
        for qtr in pf_scores['diff_jaccard']['Q1']:
            # print(qtr)
            cap_companies = []
            invested = []
            for entry in pf_scores['diff_jaccard']['Q1'][qtr]:
                # entry = [cik, score, nb_share_unbalanced, nb_share_balanced]
                cik, _, _, nb_share_balanced = entry
                price, mc, flag = post_processing.get_share_price(cik, qtr, self.lookup, self.stock_data)
                if nb_share_balanced:
                    cap_companies.append(mc)
                    invested.append(nb_share_balanced * price)
            # print("MC:", cap_companies, "Total cap:", sum(cap_companies))
            # print("Invested:", invested, "Total re-invested:", sum(invested))

            # Now that we have the cap, verify the repartition
            for idx, entry in enumerate(pf_scores['diff_jaccard']['Q1'][qtr]):
                cik, _, _, nb_share_balanced = entry
                if nb_share_balanced:
                    money_invested = invested[idx]
                    mc = cap_companies[idx]
                    # print(money_invested, sum(invested) * (mc / sum(cap_companies)))
                    # Default 7 digits check on the floats
                    self.assertAlmostEqual(money_invested, sum(invested) * (mc / sum(cap_companies)), places=2)

    def test_calculate_portfolio_value_unbalanced(self):
        """
        This rather complex, not so unitary, test validates that the unbalanced portfolio really is. It asserts that:
        (money invested in a stock) = (money available to invest)/(number of stocks invested in)
        The assertion relies on a assertAlmostEqual to the 2nd decimal place (so basically to the cent).

        :return:
        """
        # This is common between balanced and unbalanced
        tax_rate = 0
        pf_values = {m: 0 for m in self.s['metrics'][:-1]}
        for m in self.s['metrics'][:-1]:
            pf_values[m] = {q: {qtr: [0, tax_rate, 0] for qtr in self.s['list_qtr']} for q in self.s['bin_labels']}

        pf_scores = post_processing.calculate_portfolio_value(self.pf_scores, pf_values, self.lookup, self.stock_data, self.s)
        # ---------------------------------------------------

        # Now we test the unbalanced portfolio!
        for qtr in pf_scores['diff_jaccard']['Q1']:
            # print(qtr)
            cap_companies = []
            invested = []
            for entry in pf_scores['diff_jaccard']['Q1'][qtr]:
                # entry = [cik, score, nb_share_unbalanced, nb_share_balanced]
                cik, _, nb_share_unbalanced, _ = entry
                price, mc, flag = post_processing.get_share_price(cik, qtr, self.lookup, self.stock_data)
                if nb_share_unbalanced:
                    cap_companies.append(mc)
                    invested.append(nb_share_unbalanced * price)
            # print("MC:", cap_companies, "Total cap:", sum(cap_companies))
            # print("Invested:", invested, "Total re-invested:", sum(invested))

            # Now that we have the cap, verify the repartition
            for idx, entry in enumerate(pf_scores['diff_jaccard']['Q1'][qtr]):
                _, _, nb_share_unbalanced, _ = entry
                if nb_share_unbalanced:
                    money_invested = invested[idx]
                    mc = cap_companies[idx]
                    # print(money_invested, sum(invested) / len(cap_companies))
                    self.assertAlmostEqual(money_invested, sum(invested) / len(cap_companies), places=2)


if __name__ == '__main__':
    unittest.main()
