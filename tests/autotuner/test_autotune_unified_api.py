"""Unified-API dispatch tests: ``autotune(v2_opt_in=...)``.

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


def test_cache_and_v2_opt_in_are_mutually_exclusive(cache_root, tmp_path):
    with (
        pytest.raises(ValueError, match="cannot be combined with v2_opt_in"),
        autotune(True, cache=str(tmp_path / "v1.json"), v2_opt_in=True),
    ):
        pass


def test_cache_root_requires_v2_opt_in(cache_root, tmp_path):
    with (
        pytest.raises(ValueError, match="requires v2_opt_in=True"),
        autotune(True, cache_root=str(tmp_path)),
    ):
        pass


def test_v2_context_does_not_nest(cache_root):
    with (  # noqa: SIM117 - the nesting is what this test exercises
        pytest.raises(RuntimeError, match="nested v2 autotune"),
        autotune(True, v2_opt_in=True),
    ):
        with autotune(True, v2_opt_in=True):
            pass


def test_plain_autotune_still_nests_inside_managed(cache_root, monkeypatch):
    """Only the managed backend is non-nestable; plain autotune() is fine."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune(True, v2_opt_in=True):  # noqa: SIM117 - nesting is the subject
        with autotune(True):
            _, tactic = _choose()
    assert tactic == 1


def test_failed_validation_leaves_no_attachment(cache_root, tmp_path):
    """A rejected argument must not attach a store or leak context state."""
    tuner = AutoTuner.get()
    with (
        pytest.raises(ValueError),
        autotune(True, cache=str(tmp_path / "v1.json"), v2_opt_in=True),
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
        autotune(True, tuning_buckets=(), v2_opt_in=True),
    ):
        pass
    assert tuner._managed_cache is None


# --------------------------------------------------------------------------
# Dispatch reaches v2 at all, and the argument mapping is right.
#
# Deliberately thin: v2's *behaviour* is covered by test_autotune_cache_v2.py
# and is unchanged by this PR, and the v1/v2 boundary is structural (separate
# function bodies) rather than something a test can usefully police.
# --------------------------------------------------------------------------


def test_v2_opt_in_publishes_to_store(cache_root, monkeypatch):
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune(True, v2_opt_in=True):
        _, tactic = _choose()
    assert tactic == 1
    entries = _entry_files(cache_root)
    assert len(entries) == 1
    assert json.loads(entries[0].read_text())["tactic"] == 1


def test_measure_policy_requires_v2_opt_in(cache_root):
    """measure= is a v2 concept; the legacy path has no measurement policy."""
    with (
        pytest.raises(ValueError, match="measure=.*v2_opt_in=True"),
        autotune(True, measure=MeasurementPolicy(execution_mode="eager")),
    ):
        pass


# --------------------------------------------------------------------------
# autotune_v2 is now a thin alias; both spellings must agree.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opener",
    [
        lambda: autotune(True, v2_opt_in=True),
        lambda: autotune_v2(),
    ],
    ids=["dispatched", "direct"],
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
    with pytest.raises(RuntimeError, match="nested v2 autotune"), autotune_v2():  # noqa: SIM117 - nesting is the subject
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
