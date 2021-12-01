import unittest

import pandas as pd

from feedback_module import Feedback

class TestFeedbackClass(unittest.TestCase):

    def setUp(self):
        self.feedback_module = Feedback()

    def tearDown(self):
        pass

    def testGenerateAllocationFeedback(self):

        ddf = pd.DataFrame(
            {'userID': [0, 0, 1],
            'itemID': [0, 1, 0],
            'quantity': [6, 10, 4]}
        )

        fdf = pd.DataFrame(
            {'userID': [0, 2],
            'itemID': [0, 0],
            'rating': [3, 2]}
        )

        # afdf = pd.DataFrame(
        #     {'userID': [0, 0, 1],
        #     'iid_quantity': [(0, 6), (1,10), (0, 4)],
        #     'rating': [3, float('NaN'), float('NaN')]}
        # )


        afdf = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating': [3.0,]}
        )

        assert(afdf.equals(self.feedback_module.generate_allocation_feedback(ddf, fdf)))

    def testAggregate(self):
        assert(self.feedback_module.aggregate.empty)

        # six allocation feedback dataframes to feed into aggregate_feedback function
        afdf0 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating': [3]}
        )
        afdf1 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating': [4]}
        )
        afdf2 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating': [2]}
        )
        afdf3 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating': [3]}
        )
        afdf4 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating': [1]}
        )
        afdf5 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating': [5]}
        )

        # six aggregate dataframes to check against
        adf0 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating_aggregate': [3.0,],
            'rating_history': [[3],],
            }
        )
        adf1 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating_aggregate': [3.5,],
            'rating_history': [[4, 3],],
            }
        )
        adf2 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating_aggregate': [3.0,],
            'rating_history': [[2, 4, 3],],
            }
        )
        adf3 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating_aggregate': [3.0,],
            'rating_history': [[3, 2, 4, 3],],
            }
        )
        adf4 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating_aggregate': [2.6,],
            'rating_history': [[1, 3, 2, 4, 3],],
            }
        )
        adf5 = pd.DataFrame(
            {'userID': [0,],
            'iid_quantity': [(0, 6),],
            'rating_aggregate': [3.0,],
            'rating_history': [[5, 1, 3, 2, 4],],
            }
        )

        self.feedback_module.aggregate_feedback(afdf0)
        assert(adf0.equals(self.feedback_module.aggregate))
        self.feedback_module.aggregate_feedback(afdf1)
        assert(adf1.equals(self.feedback_module.aggregate))
        self.feedback_module.aggregate_feedback(afdf2)
        assert(adf2.equals(self.feedback_module.aggregate))
        self.feedback_module.aggregate_feedback(afdf3)
        assert(adf3.equals(self.feedback_module.aggregate))
        self.feedback_module.aggregate_feedback(afdf4)
        assert(adf4.equals(self.feedback_module.aggregate))
        self.feedback_module.aggregate_feedback(afdf5)
        assert(adf5.equals(self.feedback_module.aggregate))

    def testGeneratePredictions(self):
        assert(self.feedback_module.aggregate.empty)

        # decision df
        ddf = pd.DataFrame(
            {'userID': [0, 0, 0, 1, 1, 1, 2, 2, 2,],
            'itemID': [0, 1, 2, 0, 1, 2, 0, 1, 2,],
            'quantity': [4, 9, 20, 4, 9, 10, 4, 9, 30],
            }
        )

        # feedback df
        fdf = pd.DataFrame(
            {'userID': [0, 0, 1, 1, 2, 2, 2,],
            'itemID': [0, 2, 0, 2, 1, 2, 2,],
            'rating': [4, 2, 5, 5, 1, 4, 2,],
            }
        )

        # allocation feedback df
        afdf = pd.DataFrame(
            {'userID': [0, 0, 1, 1, 1, 2, 2, 2],
            'iid_quantity': [(0, 4),(2, 30), (0, 4), (1, 9), (2, 30), (1,9), (2,20), (2,30)],
            'rating_aggregate': [3.8, 2.5, 3.75, 2.4, 4.5, 1.0, 4.0, 2.0,],
            'rating_history': [[1, 5, 5, 4, 4],[3, 2], [2, 4, 4, 5], [2, 2, 3, 2, 3], [4, 5], [1, ], [4, 4, 4, 4, 4], [2, 2],],
            }
        )

        print (afdf)
        print (ddf)
        # feedback prediction df
        fpdf = self.feedback_module.generate_predictions(afdf, ddf, fdf)
        print (fpdf)

if __name__ == '__main__':
    unittest.main()