# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler.config import ProfilerConfig, TorchProfilerToolConfig
from verl.utils.profiler.profile import DistProfiler, _NoOpProfiler
from verl.utils.profiler.torch_profile import (
    Profiler,
    build_trace_basename,
    get_torch_profiler,
)


class TestTorchProfile(unittest.TestCase):
    def setUp(self):
        # Reset process-global Profiler class state so tests don't leak into each other.
        Profiler._define_count = 0
        Profiler._active_prof = None
        Profiler._owner = None

    def tearDown(self):
        Profiler._define_count = 0
        Profiler._active_prof = None
        Profiler._owner = None

    @patch("torch.profiler.profile")
    def test_get_torch_profiler(self, mock_profile):
        # Test wrapper function
        get_torch_profiler(contents=["cpu", "cuda", "stack"], save_path="/tmp/test", rank=0)
        mock_profile.assert_called_once()
        _, kwargs = mock_profile.call_args

        # Verify activities
        activities = kwargs["activities"]
        self.assertIn(torch.profiler.ProfilerActivity.CPU, activities)
        self.assertIn(torch.profiler.ProfilerActivity.CUDA, activities)

        # Verify options
        self.assertTrue(kwargs["with_stack"])
        self.assertFalse(kwargs["record_shapes"])
        self.assertFalse(kwargs["profile_memory"])

    @patch("torch.profiler.profile")
    def test_role_goes_in_filename_not_a_directory(self, mock_profile):
        # The role must not create a directory level: one step's traces would be scattered
        # across sibling dirs and hidden from finish_hook_cmd, which only sees save_path.
        with tempfile.TemporaryDirectory() as save_path:
            get_torch_profiler(contents=["cpu"], save_path=save_path, role="train", save_file_prefix="actor", rank=0)
            # No "train" sub-directory is created next to the traces.
            self.assertEqual(os.listdir(save_path), [])

            _, kwargs = mock_profile.call_args
            mock_prof = MagicMock()
            kwargs["on_trace_ready"](mock_prof)

            (out_path,), _ = mock_prof.export_chrome_trace.call_args
            self.assertEqual(os.path.dirname(out_path), save_path)
            self.assertTrue(os.path.basename(out_path).startswith("actor_train_"))
            self.assertTrue(out_path.endswith(".json.gz"))

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_profiler_lifecycle(self, mock_get_profiler):
        # Mock the underlying torch profiler object
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        # Initialize
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # Test Start
        profiler.start()
        mock_get_profiler.assert_called_once()
        mock_prof_instance.start.assert_called_once()

        # Test Step
        profiler.step()
        mock_prof_instance.step.assert_called_once()

        # Test Stop
        profiler.stop()
        mock_prof_instance.stop.assert_called_once()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_start_forwards_training_step(self, mock_get_profiler):
        # The trainer reports the profiled step as start_profile(profile_step=...); it must reach
        # the filename, otherwise traces from different steps cannot be told apart.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        profiler.start(role="train", profile_step=3)
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["profile_step"], 3)
        self.assertEqual(kwargs["role"], "train")
        profiler.stop()

    @patch("torch.profiler.record_function")
    def test_continuous_annotate_range_names_role_and_function(self, mock_record_function):
        # Continuous mode puts every stage in one trace, so the range name has to carry the role as
        # well as the function: a bare "compute_log_prob" cannot say whether the forward belonged to
        # the actor or to the reference model colocated in the same process.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        @profiler.annotate(role="ref_compute_log_prob")
        def compute_log_prob():
            return "done"

        @profiler.annotate()
        def train_batch():
            return "done"

        self.assertEqual(compute_log_prob(), "done")
        mock_record_function.assert_called_with("ref_compute_log_prob")

        # A stage that declares no role is still named after the function it wraps.
        self.assertEqual(train_batch(), "done")
        mock_record_function.assert_called_with("train_batch")

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_annotate_names_stage_and_step(self, mock_get_profiler):
        # Discrete mode opens one profiler per stage from annotate(), long after start(): the
        # stage name and the training step recorded at start() must both reach the filename.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)
        profiler.start(role="train", profile_step=5)

        @profiler.annotate(role="actor_update")
        def update_actor():
            return "done"

        # An unlabeled stage falls back to the wrapped function's name.
        @profiler.annotate()
        def compute_log_prob():
            return "done"

        self.assertEqual(update_actor(), "done")
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["role"], "actor_update")
        self.assertEqual(kwargs["profile_step"], 5)

        self.assertEqual(compute_log_prob(), "done")
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["role"], "compute_log_prob")

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_mode(self, mock_get_profiler):
        # Mock for discrete mode
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # In discrete mode, start/stop shouldn't trigger global profiler immediately
        profiler.start()
        mock_get_profiler.assert_not_called()

        profiler.stop()
        mock_prof_instance.stop.assert_not_called()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_annotate_stops_profiler_on_exception(self, mock_get_profiler):
        # A stage raising inside a discrete-mode annotate must still stop the
        # (process-global) torch profiler; otherwise it leaks, the next stage's
        # start() fails with "Profiler is already enabled" and the process aborts.
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        calls = {"n": 0}

        @profiler.annotate(role="boom")
        def boom():
            calls["n"] += 1
            raise RuntimeError("stage failed on purpose")

        with self.assertRaises(RuntimeError):
            boom()

        # Profiler must be started and, crucially, stopped despite the exception,
        # and the stage body must run exactly once (no re-execution).
        mock_prof_instance.start.assert_called_once()
        mock_prof_instance.stop.assert_called_once()
        self.assertEqual(calls["n"], 1)

    def test_dist_annotate_propagates_and_runs_func_once(self):
        # DistProfiler.annotate must not swallow errors from the wrapped function
        # nor re-run it (which would execute the stage twice). It only falls back
        # when *setting up* backend profiling fails.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )

        class PassthroughImpl:
            def annotate(self, **kwargs):
                def decorator(fn):
                    return fn

                return decorator

        calls = {"n": 0}

        class FakeWorker:
            def __init__(self, profiler):
                self.profiler = profiler
                self.rank = 0

            @DistProfiler.annotate(role="boom")
            def boom(self):
                calls["n"] += 1
                raise RuntimeError("stage failed on purpose")

        dp = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dp._impl = PassthroughImpl()
        dp._this_step = True  # simulate an active profiled step

        worker = FakeWorker(dp)
        with self.assertRaises(RuntimeError):
            worker.boom()
        self.assertEqual(calls["n"], 1)

    def test_dist_annotate_falls_back_when_setup_fails(self):
        # If backend annotate setup raises, the function still runs (once), unprofiled.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )

        class BadImpl:
            def annotate(self, **kwargs):
                raise RuntimeError("cannot set up profiling")

        calls = {"n": 0}

        class FakeWorker:
            def __init__(self, profiler):
                self.profiler = profiler
                self.rank = 0

            @DistProfiler.annotate(role="x")
            def do_work(self):
                calls["n"] += 1
                return "ok"

        dp = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dp._impl = BadImpl()
        dp._this_step = True

        worker = FakeWorker(dp)
        self.assertEqual(worker.do_work(), "ok")
        self.assertEqual(calls["n"], 1)

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_collection_is_never_sub_sampled(self, mock_profile, mock_schedule):
        # A profiled step is collected whole: no torch.profiler.schedule, so nothing can drop the
        # stages that run before the update loop (which is what sub-step sampling used to do).
        get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0)

        mock_schedule.assert_not_called()
        _, kwargs = mock_profile.call_args
        self.assertNotIn("schedule", kwargs)

    @patch("torch.profiler.profile")
    def test_second_flush_does_not_overwrite_the_first(self, mock_profile):
        # Both windows share one filename stem, so the second needs a suffix of its own.
        with tempfile.TemporaryDirectory() as save_path:
            get_torch_profiler(contents=["cpu"], save_path=save_path, rank=0)
            on_trace_ready = mock_profile.call_args[1]["on_trace_ready"]

            names = []
            for _ in range(2):
                mock_prof = MagicMock()
                on_trace_ready(mock_prof)
                (out_path,), _ = mock_prof.export_chrome_trace.call_args
                names.append(os.path.basename(out_path))

            self.assertNotIn("_part", names[0])
            self.assertTrue(names[1].endswith("_part1.json.gz"), names[1])

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_only_the_owning_profiler_closes_a_step(self, mock_get_profiler):
        # A colocated actor and reference model each own a Profiler but share one process-global
        # collection, and the trainer reports the end of a training step to both. Marking it twice
        # would show two step boundaries where one training step ended.
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        actor = Profiler(rank=0, config=config, tool_config=tool_config)
        colocated_ref = Profiler(rank=0, config=config, tool_config=tool_config)

        actor.start(role="train", profile_step=1)
        colocated_ref.start(profile_step=1)
        # One collection for the process, opened by the first role to start.
        mock_get_profiler.assert_called_once()

        actor.step()
        colocated_ref.step()
        self.assertEqual(mock_prof_instance.step.call_count, 1)

        # stop() closes the last step's window before tearing the collection down.
        actor.stop()
        self.assertEqual(mock_prof_instance.step.call_count, 2)
        mock_prof_instance.stop.assert_called_once()
        self.assertIsNone(Profiler._active_prof)

    def test_build_trace_basename_encodes_role_rank_and_parallelism(self):
        # Filename stem must embed the worker role, scope role, rank/world size and the
        # tp/pp/dp/cp parallel ranks so per-process traces are self-describing.
        name = build_trace_basename(
            rank=5,
            role="train",
            save_file_prefix="actor",
            topology={"rank": 5, "world_size": 16, "tp": 1, "pp": 0, "dp": 2, "cp": 0},
        )
        self.assertTrue(name.startswith("actor_train_"))
        self.assertIn("rank5-of-16", name)
        self.assertIn("tp1-pp0-dp2-cp0", name)
        self.assertIn(f"pid{os.getpid()}", name)

    def test_build_trace_basename_encodes_training_step(self):
        # Without the step, traces from different profiled steps are only distinguishable by
        # their wall-clock timestamp, which is unreadable when several steps are profiled.
        name = build_trace_basename(
            rank=0, role="train", save_file_prefix="actor", profile_step=7, topology={"rank": 0, "world_size": 8}
        )
        self.assertTrue(name.startswith("actor_train_step7_rank0-of-8_"))

    def test_build_trace_basename_omits_step_when_unknown(self):
        name = build_trace_basename(rank=0, role="train", save_file_prefix="actor", topology={})
        self.assertTrue(name.startswith("actor_train_rank0_"))
        self.assertNotIn("step", name)

    def test_build_trace_basename_distinguishes_roles_same_rank(self):
        # The original scheme collided ref/critic at the same rank; the role prefix fixes it.
        topo = {"rank": 5, "world_size": 16}
        ref_name = build_trace_basename(rank=5, save_file_prefix="ref", topology=topo)
        critic_name = build_trace_basename(rank=5, save_file_prefix="value_model", topology=topo)
        self.assertTrue(ref_name.startswith("ref_rank5-of-16_"))
        # Underscores in labels are normalized to hyphens (underscore is the field separator).
        self.assertTrue(critic_name.startswith("value-model_rank5-of-16_"))
        self.assertNotEqual(ref_name, critic_name)

    def test_build_trace_basename_minimal_topology(self):
        # With no distributed topology, fall back to the passed rank and omit parallel dims.
        name = build_trace_basename(rank=3, topology={})
        self.assertTrue(name.startswith("rank3_"))
        self.assertNotIn("-of-", name)
        for dim in ("tp", "pp", "dp", "cp"):
            self.assertNotIn(f"{dim}0", name)

    def test_build_trace_basename_sanitizes_labels(self):
        # Slashes/spaces in labels must not leak into the filename.
        name = build_trace_basename(rank=0, role="update actor", save_file_prefix="actor/rollout", topology={})
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)
        self.assertIn("actor-rollout", name)
        self.assertIn("update-actor", name)

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_dist_profiler_forwards_save_file_prefix(self, mock_get_profiler):
        # DistProfiler must forward save_file_prefix down to the torch backend so it
        # ends up in the trace filename.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config, save_file_prefix="actor")
        self.assertEqual(dist_profiler._impl.save_file_prefix, "actor")

        dist_profiler.start()
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["save_file_prefix"], "actor")
        dist_profiler.stop()

    def test_dist_profiler_step_noop_backend(self):
        # A backend without scheduling support (no-op impl) must make step() a safe no-op.
        config = ProfilerConfig(tool=None, enable=True, all_ranks=True, save_path="/tmp/test", tool_config=None)
        dist_profiler = DistProfiler(rank=0, config=config)
        self.assertIsNone(dist_profiler.step())

    def test_dist_profiler_step_disabled(self):
        # When disabled, step() must not touch the backend at all.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=False, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dist_profiler._impl = MagicMock()
        self.assertIsNone(dist_profiler.step())
        dist_profiler._impl.step.assert_not_called()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_dist_profiler_step_torch_delegates(self, mock_get_profiler):
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)

        dist_profiler.start()
        dist_profiler.step()
        mock_prof_instance.step.assert_called_once()

        dist_profiler.stop()


class TestCpuActivityAlwaysCollected(unittest.TestCase):
    """Stage markers are CPU-side events, so a device-only trace hides every verl stage.

    ``contents`` is left exactly as the user wrote it; CPU activity is enabled when the profiler is
    built, so no configuration can produce a trace of unattributable kernels.
    """

    @patch("torch.profiler.profile")
    def test_cpu_activity_added_for_device_only_contents(self, mock_profile):
        for contents in (["cuda"], ["cuda", "memory"], []):
            with self.subTest(contents=contents):
                mock_profile.reset_mock()
                get_torch_profiler(contents=list(contents), save_path="/tmp/test", rank=0)
                activities = mock_profile.call_args[1]["activities"]
                self.assertIn(torch.profiler.ProfilerActivity.CPU, activities)
                self.assertIn(torch.profiler.ProfilerActivity.CUDA, activities)

    @patch("torch.profiler.profile")
    def test_contents_are_not_rewritten(self, mock_profile):
        # The user's selection is theirs: enabling the activity must not edit the config.
        contents = ["cuda"]
        tool_config = TorchProfilerToolConfig(contents=contents)
        get_torch_profiler(contents=tool_config.contents, save_path="/tmp/test", rank=0)
        self.assertEqual(tool_config.contents, ["cuda"])
        self.assertEqual(contents, ["cuda"])


def _role_profiler_omegaconf(tool="torch", enable=True, discrete=False, contents=("cpu", "cuda")):
    """Mimic a per-role ``profiler`` OmegaConf sub-tree (identical across ref/ref.yaml and
    critic/critic.yaml).

    The nested ``_target_`` entries are what the hydra instantiation path (omega_conf_to_dataclass
    without an explicit dataclass_type) uses to build real dataclass tool configs, as opposed to the
    plain dicts the torch profiler cannot consume.
    """
    return OmegaConf.create(
        {
            "_target_": "verl.utils.profiler.ProfilerConfig",
            "tool": tool,
            "enable": enable,
            "all_ranks": False,
            "ranks": [0],
            "save_path": "/tmp/test_role_profile",
            "tool_config": {
                "torch": {
                    "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                    "contents": list(contents),
                    "discrete": discrete,
                },
            },
        }
    )


class TestRefWorkerProfilerConfig(unittest.TestCase):
    """The reference model's inner TrainingWorker must receive a real, torch-consumable profiler
    config (mirroring the actor), instead of silently running with a disabled no-op profiler.

    ``ActorRolloutRefWorker.init_model`` now forwards the ref's own ``profiler`` config to the ref
    ``TrainingWorkerConfig`` via ``omega_conf_to_dataclass(self.config.ref.get("profiler", {}))``.
    These lock in that conversion path so a torch profiler config actually yields a torch backend on
    the ref worker, while an absent config degrades to a no-op (the previous ref behavior).
    """

    def test_ref_profiler_config_builds_torch_backend(self):
        # Exercise the exact conversion init_model performs on actor_rollout_ref.ref.profiler:
        # omega_conf_to_dataclass(...) (no dataclass_type) must resolve the _target_ entries into
        # real nested dataclasses the torch Profiler can consume via attribute access.
        omega_cfg = _role_profiler_omegaconf(tool="torch", enable=True)
        ref_profiler_config = omega_conf_to_dataclass(omega_cfg)

        self.assertIsInstance(ref_profiler_config, ProfilerConfig)
        self.assertTrue(ref_profiler_config.enable)
        self.assertEqual(ref_profiler_config.tool, "torch")

        # TrainingWorker.__init__ extracts the tool-specific config exactly like this; it must be a
        # real dataclass (not a plain dict) for the torch Profiler to read .contents/.discrete.
        tool_config = ref_profiler_config.tool_config.get(ref_profiler_config.tool)
        self.assertIsInstance(tool_config, TorchProfilerToolConfig)
        self.assertEqual(tool_config.contents, ["cpu", "cuda"])

        dist_profiler = DistProfiler(
            rank=0, config=ref_profiler_config, tool_config=tool_config, save_file_prefix="ref"
        )
        self.assertIsInstance(dist_profiler._impl, Profiler)
        self.assertTrue(dist_profiler.check_enable())
        self.assertTrue(dist_profiler.check_this_rank())

    def test_absent_ref_profiler_config_is_disabled_noop(self):
        # Contrast with the previous behavior: without a profiler_config the ref worker built a
        # disabled no-op profiler, so the reference model was never profiled by its own worker.
        dist_profiler = DistProfiler(rank=0, config=None)
        self.assertIsInstance(dist_profiler._impl, _NoOpProfiler)
        self.assertFalse(dist_profiler.check_enable())


class TestCriticWorkerProfilerConfig(unittest.TestCase):
    """The critic is a standalone TrainingWorker (no outer ActorRolloutRefWorker wrapper): the
    trainer drives start_profile()/stop_profile() and the ``train_batch`` annotation directly on it.

    ``RayPPOTrainer._init_workers`` (and the v1 / separation trainer variants) now forward
    ``omega_conf_to_dataclass(self.config.critic.get("profiler", {}))`` into the critic
    ``TrainingWorkerConfig``. Without it the critic's DistProfiler silently degraded to a no-op, so
    the critic was never profiled by any backend. These lock in that wiring.
    """

    def test_critic_profiler_config_builds_torch_backend(self):
        # critic/critic.yaml's profiler block is structurally identical to ref/ref.yaml; the trainer
        # converts it the same way. It must yield a real torch backend on the standalone critic worker.
        omega_cfg = _role_profiler_omegaconf(tool="torch", enable=True)
        critic_profiler_config = omega_conf_to_dataclass(omega_cfg)

        self.assertIsInstance(critic_profiler_config, ProfilerConfig)
        tool_config = critic_profiler_config.tool_config.get(critic_profiler_config.tool)
        self.assertIsInstance(tool_config, TorchProfilerToolConfig)

        # The critic TrainingWorker uses model_type="value_model" as the trace filename prefix.
        dist_profiler = DistProfiler(
            rank=0, config=critic_profiler_config, tool_config=tool_config, save_file_prefix="value_model"
        )
        self.assertIsInstance(dist_profiler._impl, Profiler)
        self.assertTrue(dist_profiler.check_enable())
        self.assertTrue(dist_profiler.check_this_rank())
        self.assertEqual(dist_profiler._impl.save_file_prefix, "value_model")

    def test_absent_critic_profiler_config_is_disabled_noop(self):
        # The previous behavior: the trainer built the critic TrainingWorkerConfig without a
        # profiler_config, so DistProfiler(config=None) degraded to a disabled no-op.
        dist_profiler = DistProfiler(rank=0, config=None)
        self.assertIsInstance(dist_profiler._impl, _NoOpProfiler)
        self.assertFalse(dist_profiler.check_enable())


if __name__ == "__main__":
    unittest.main()
