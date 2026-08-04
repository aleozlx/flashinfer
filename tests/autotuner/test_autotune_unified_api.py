"""Unified-API dispatch tests: ``autotune(managed_cache=...)``.

The managed store and the legacy JSON file cache are two persistence
backends behind one entry point.  These tests pin the property that makes
that safe: the backend is chosen once, on context entry, and the two can
never interleave within a context.  See ``docs/design_docs/autotuner_v2.md``
§2.1 and §5.1.

GPU-free: profiling is monkeypatched to a lookup table.
"""

import json

import pytest
import torch

from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2
from flashinfer.autotuner import AutoTuner, TuningConfig, autotune

from .utils import DummyRunner, reset_autotuner

_OP = "test::autotune_unified"
_CONFIG = TuningConfig()


def _fresh_process():
    tuner = reset_autotuner()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner._managed_decoded.clear()
    return tuner


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    _fresh_process()
    yield tmp_path
    _fresh_process()


def _install_fake_profile(monkeypatch, times):
    calls = []

    def fake_profile(self, runner, inputs, tactic, tuning_config, **kwargs):
        calls.append(tactic)
        return float(times[tactic])

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)
    return calls


def _choose(tactics=(0, 1, 2)):
    return AutoTuner.get().choose_one(
        _OP, [DummyRunner(tactics)], _CONFIG, [torch.zeros(8, 16)]
    )


def _entry_files(cache_root):
    return list(cache_root.glob("v2/*/entries/*.json"))


# --------------------------------------------------------------------------
# Argument validation: the bifurcation is explicit and cannot be ambiguous.
# --------------------------------------------------------------------------


def test_cache_and_managed_cache_are_mutually_exclusive(cache_root, tmp_path):
    with (
        pytest.raises(ValueError, match="different persistence backends"),
        autotune(True, cache=str(tmp_path / "v1.json"), managed_cache=True),
    ):
        pass


def test_cache_root_requires_managed_cache(cache_root, tmp_path):
    with (
        pytest.raises(ValueError, match="requires managed_cache=True"),
        autotune(True, cache_root=str(tmp_path)),
    ):
        pass


def test_managed_cache_does_not_nest(cache_root):
    with (  # noqa: SIM117 - the nesting is what this test exercises
        pytest.raises(RuntimeError, match="nested managed-cache"),
        autotune(True, managed_cache=True),
    ):
        with autotune(True, managed_cache=True):
            pass


def test_plain_autotune_still_nests_inside_managed(cache_root, monkeypatch):
    """Only the managed backend is non-nestable; plain autotune() is fine."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune(True, managed_cache=True):  # noqa: SIM117 - nesting is the subject
        with autotune(True):
            _, tactic = _choose()
    assert tactic == 1


def test_failed_validation_leaves_no_attachment(cache_root, tmp_path):
    """A rejected argument must not attach a store or leak context state."""
    tuner = AutoTuner.get()
    with (
        pytest.raises(ValueError),
        autotune(True, cache=str(tmp_path / "v1.json"), managed_cache=True),
    ):
        pass
    assert tuner._managed_cache is None
    assert not tuner._v2_local.active
    assert not _entry_files(cache_root)


def test_empty_buckets_still_raise_before_attaching(cache_root):
    """Legacy validation runs before the managed attach, so it wins."""
    tuner = AutoTuner.get()
    with (
        pytest.raises(ValueError, match="tuning_buckets"),
        autotune(True, tuning_buckets=(), managed_cache=True),
    ):
        pass
    assert tuner._managed_cache is None


# --------------------------------------------------------------------------
# Dispatch: each backend does its own thing and only its own thing.
# --------------------------------------------------------------------------


def test_managed_cache_true_publishes_to_store(cache_root, monkeypatch):
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune(True, managed_cache=True):
        _, tactic = _choose()
    assert tactic == 1
    entries = _entry_files(cache_root)
    assert len(entries) == 1
    assert json.loads(entries[0].read_text())["tactic"] == 1


def test_default_is_byte_identical_legacy(cache_root, monkeypatch, tmp_path):
    """No managed_cache argument -> the v1 path, untouched: a JSON file and
    no managed directory anywhere."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    path = tmp_path / "v1.json"
    with autotune(True, cache=str(path)):
        _choose()
    assert path.is_file()
    assert not _entry_files(cache_root)
    assert AutoTuner.get()._managed_cache is None


def test_managed_cache_writes_no_v1_file(cache_root, monkeypatch):
    """The managed branch must not populate v1's in-memory file configs."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    tuner = AutoTuner.get()
    with autotune(True, managed_cache=True):
        _choose()
    assert tuner._file_configs == {}


def test_managed_cache_false_forbids_disk(cache_root, monkeypatch):
    """managed_cache=False tunes in memory even though a store is attached."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune(True, managed_cache=True):
        _choose()
    assert len(_entry_files(cache_root)) == 1

    reset_autotuner()  # keeps the attached store, drops tuned winners
    _install_fake_profile(monkeypatch, times={0: 1.0, 1: 3.0, 2: 2.0})
    with autotune(True, managed_cache=False):
        _, tactic = _choose()
    assert tactic == 0  # freshly measured, not the stored winner
    assert len(_entry_files(cache_root)) == 1  # nothing new published


def test_attach_survives_context_exit(cache_root, monkeypatch):
    """managed_cache=True attaches for the process: serving afterwards, with
    no context at all, still resolves to the store."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune(True, managed_cache=True):
        _choose()

    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    reset_autotuner()
    _, tactic = _choose()  # bare, outside any context
    assert tactic == 1
    assert calls == []  # served from the store, never profiled


def test_replay_mode_does_not_profile(cache_root, monkeypatch):
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune(True, managed_cache=True):
        _choose()

    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    reset_autotuner()
    with autotune(False, managed_cache=True):
        _, tactic = _choose()
    assert tactic == 1
    assert calls == []


def test_measure_policy_without_managed_cache(cache_root, monkeypatch):
    """A measurement policy is independent of persistence: it applies without
    managed_cache=True, and publishes nothing."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune(True, measure=MeasurementPolicy(execution_mode="eager")):
        _, tactic = _choose()
    assert tactic == 1
    assert not _entry_files(cache_root)
    assert AutoTuner.get()._managed_cache is None


# --------------------------------------------------------------------------
# autotune_v2 is now a thin alias; both spellings must agree.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opener",
    [
        lambda: autotune(True, managed_cache=True),
        lambda: autotune_v2(),
        lambda: autotune_v2(mode="tune", persistent_cache=True),
    ],
    ids=["unified", "alias_default", "alias_explicit"],
)
def test_alias_and_unified_agree(cache_root, monkeypatch, opener):
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with opener():
        _, tactic = _choose()
    assert tactic == 1
    assert len(_entry_files(cache_root)) == 1


def test_alias_replay_maps_to_tune_mode_false(cache_root, monkeypatch):
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2():
        _choose()

    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    reset_autotuner()
    with autotune_v2(mode="replay"):
        _, tactic = _choose()
    assert tactic == 1
    assert calls == []


def test_alias_rejects_nesting_via_unified_guard(cache_root):
    with pytest.raises(RuntimeError, match="nested managed-cache"), autotune_v2():  # noqa: SIM117 - nesting is the subject
        with autotune_v2():
            pass


def test_alias_still_validates_its_own_arguments(cache_root):
    with pytest.raises(ValueError, match="mode must be"), autotune_v2(mode="nonsense"):
        pass
    with (
        pytest.raises(TypeError, match="persistent_cache must be a bool"),
        autotune_v2(persistent_cache="/tmp/somewhere"),
    ):
        pass
