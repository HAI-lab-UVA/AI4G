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

        assert(afdf.equals(self.feedback_module.generate_item_allocation_feedback(ddf, fdf)))


if __name__ == '__main__':
    unittest.main()