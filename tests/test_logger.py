import os
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

import unittest

from soccertrack.logger import *


class TestLogger(unittest.TestCase):
    def test_name_eq_main(self):
        """Test that the if __name__ == "__main__" block executes without
        error."""
        loader = SourceFileLoader(
            "__main__",
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "soccertrack", "logger.py"
            ),
        )
        loader.exec_module(module_from_spec(spec_from_loader(loader.name, loader)))

    def test_set_log_level(self):
        set_log_level("DEBUG")
        self.assertEqual(level_filter.level, "DEBUG")
        self.assertEqual(os.environ["LOG_LEVEL"], "DEBUG")

    def test_tqdm(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "test.txt", "w") as f:
                for i in tqdm(range(10), level="DEBUG"):
                    f.write(str(i))
            self.assertTrue((tmpdir / "test.txt").exists())

    def test_inspect(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / "test.txt", "w") as f:
                inspect(f, level="DEBUG")
            self.assertTrue((tmpdir / "test.txt").exists())

    def show_df(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        inspect(df, level="DEBUG")
