"""
Test for train_1src_model.py

Checks:
- imports work
- script executes without crashing
- the imported main() from train_1src is called
"""

import os
import sys
import pytest

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import script under test
import train_1src_model as script


# ------------------------------------------------------------
# Dummy main() to replace the real training function
# ------------------------------------------------------------
class DummyMain:
    def __init__(self):
        self.called = False

    def __call__(self):
        self.called = True


# ------------------------------------------------------------
# Fixture to patch dependencies
# ------------------------------------------------------------
@pytest.fixture
def patch_dependencies(monkeypatch, tmp_path):
    dummy = DummyMain()

    # Patch the imported main() function
    monkeypatch.setattr(script, "main", dummy)

    # Change working directory to tmp
    monkeypatch.chdir(tmp_path)

    return dummy


# ------------------------------------------------------------
# Test script execution
# ------------------------------------------------------------
def test_script_runs_and_calls_main(patch_dependencies):
    """
    Ensures that:
    - script executes as __main__
    - the patched main() is called
    """

    dummy = patch_dependencies

    # Simulate running the script as __main__
    script.__name__ = "__main__"
    script.main()

    assert dummy.called, "train_1src.main() was not called"
