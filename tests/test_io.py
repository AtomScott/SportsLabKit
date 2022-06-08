
import os
import sys
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from sandbox.io import (
    auto_string_parser,
    save_dataframe,
    load_dataframe
)

import unittest


class TestIO(unittest.TestCase):
    
    def test_auto_string_parser(self):
        
        for value in [True, False, 500, 50.0, "Hellow World"]:
            s = f"{value}"
            parsed_value = auto_string_parser(s)
            self.assertEqual(parsed_value, value)
    
    def test_save_dataframe_load_dataframe(self):
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / Path("dataframe.csv")
            
            # Make a dataframe with multi-index columns
            cols = pd.MultiIndex.from_tuples([
                ('0', '0', 'A'),
                ('0', '0', 'B'),
            ])
            dataframe = pd.DataFrame(
                data=[[1, 2], [3, 4]],
                index=['a', 'b'],
                columns=cols
            )
            
            dataframe.attrs = {
                'testinging': 'test',
                '420': 420,
                '2.6': 2.6
            }
            
            save_dataframe(dataframe, save_path)
            self.assertTrue(save_path.exists())

            loaded_dataframe = load_dataframe(save_path)

            pd.testing.assert_frame_equal(loaded_dataframe, dataframe)
            self.assertDictEqual(
                loaded_dataframe.attrs,
                dataframe.attrs
            )

if __name__ == "__main__":
    unittest.main()
