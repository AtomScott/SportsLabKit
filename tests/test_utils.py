import unittest
from pathlib import Path

from soccertrack.utils.utils import (
    get_git_root
)

class TestUtils(unittest.TestCase):
    def test_get_git_root(self):
        git_root = get_git_root()
        self.assertEqual(
            Path(git_root).absolute(),
            Path(__file__).parent.parent.absolute()
            )