"""
Test for make_surrogate_dataset.py

Checks:
- imports work
- main() runs without crashing
- SurrogateDataGenerator.generate() is called correctly
- SurrogateDataGenerator.save_npz() is called correctly
- report_dataset() loads npz files correctly
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import the script under test
import make_surrogate_dataset as script


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------
class DummyPhysics:
    """Mock PhysicsTM"""
    def __init__(self):
        pass


class DummyGenerator:
    """Mock SurrogateDataGenerator"""

    def __init__(self, physics, output_dir, num_angles, rho_min, rho_max):
        self.physics = physics
        self.output_dir = output_dir
        self.num_angles = num_angles
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.calls_generate = []
        self.calls_save = []

    def generate(self, kind, n_sources):
        self.calls_generate.append((kind, n_sources))
        # Return dummy arrays
        X = np.zeros((10, 5), dtype=np.float32)
        Y = np.zeros((10, 2), dtype=np.float32)
        return X, Y

    def save_npz(self, X, Y, filename):
        self.calls_save.append(filename)
        # Create a dummy npz file in a temp directory
        np.savez(filename, X=X, Y=Y)


# ------------------------------------------------------------
# Monkeypatch imports inside the script
# ------------------------------------------------------------
@pytest.fixture
def patch_dependencies(monkeypatch, tmp_path):
    # Patch PhysicsTM
    monkeypatch.setattr(
        script,
        "PhysicsTM",
        lambda: DummyPhysics()
    )

    # Patch SurrogateDataGenerator
    def dummy_gen_factory(**kwargs):
        return DummyGenerator(**kwargs)

    monkeypatch.setattr(
        script,
        "SurrogateDataGenerator",
        dummy_gen_factory
    )

    # Patch report_dataset to skip printing
    monkeypatch.setattr(
        script,
        "report_dataset",
        lambda path: None
    )

    # Change working directory to tmp
    monkeypatch.chdir(tmp_path)

    return tmp_path


# ------------------------------------------------------------
# Test main()
# ------------------------------------------------------------
def test_main_runs_without_errors(patch_dependencies, monkeypatch):
    """
    Ensures that:
    - main() executes
    - generate() is called twice (Esurf, Hsurf)
    - save_npz() is called twice
    """

    # Fake argv so argparse does NOT see pytest args
    monkeypatch.setattr(sys, "argv", ["make_surrogate_dataset.py"])

    # Capture calls by injecting a generator instance
    calls = {"gen": None}

    def dummy_gen_factory(physics, output_dir, num_angles, rho_min, rho_max):
        gen = DummyGenerator(
            physics=physics,
            output_dir=output_dir,
            num_angles=num_angles,
            rho_min=rho_min,
            rho_max=rho_max
        )
        calls["gen"] = gen
        return gen

    monkeypatch.setattr(script, "SurrogateDataGenerator", dummy_gen_factory)

    # Run main()
    script.main()

    gen = calls["gen"]
    assert gen is not None, "Generator was not instantiated"

    # Check generate calls
    assert len(gen.calls_generate) == 2
    assert gen.calls_generate[0][0] == "Esurf"
    assert gen.calls_generate[1][0] == "Hsurf"

    # Check save calls
    assert len(gen.calls_save) == 2
    assert "Esurf.npz" in gen.calls_save[0]
    assert "Hsurf.npz" in gen.calls_save[1]


# ------------------------------------------------------------
# Test report_dataset()
# ------------------------------------------------------------
def test_report_dataset_loads_npz_correctly(tmp_path):
    """
    Creates a dummy npz file and checks that report_dataset()
    loads X and Y without errors.
    """

    # Create dummy npz
    X = np.zeros((10, 5), dtype=np.float32)
    Y = np.zeros((10, 2), dtype=np.float32)
    path = tmp_path / "dummy.npz"
    np.savez(path, X=X, Y=Y)

    # Should not raise
    script.report_dataset(str(path))
