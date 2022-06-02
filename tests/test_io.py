
import os
import sys
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
    
    def test_save_dataframe(self):
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / Path("dataframe.csv")
            
    
    def test_load_dataframe(self):
    
    

if __name__ == "__main__":
    unittest.main()
