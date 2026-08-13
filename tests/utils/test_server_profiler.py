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

import json
import os
import tempfile
import unittest
from functools import partial
from unittest.mock import AsyncMock, MagicMock, patch

from verl.utils.profiler.config import (
    NPUToolConfig,
    ProfilerConfig,
    TorchProfilerToolConfig,
    build_sglang_profiler_args,
    build_vllm_profiler_args,
    relocate_rollout_traces,
    rollout_trace_dir,
)
from verl.utils.profiler.profile import DistProfiler, build_rollout_dist_profiler


class TestServerProfilerArgs(unittest.TestCase):
    def test_build_vllm_profiler_args(self):
        # Case 1: All features enabled
        tool_config = TorchProfilerToolConfig(contents=["stack", "shapes", "memory"])
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)

        # Patch environ to avoid side effects and verify calls
        with patch.dict(os.environ, {}, clear=True):
            args = build_vllm_profiler_args(config, tool_config, rank=0)

            # Check Env vars (backward compatibility)
            self.assertEqual(os.environ.get("VLLM_TORCH_PROFILER_DIR"), "/tmp/test/agent_loop_rollout_replica_0")
            self.assertEqual(os.environ.get("VLLM_TORCH_PROFILER_WITH_STACK"), "1")
            self.assertEqual(os.environ.get("VLLM_TORCH_PROFILER_RECORD_SHAPES"), "1")
            self.assertEqual(os.environ.get("VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"), "1")

            # Check Args (new API)
            self.assertIn("profiler_config", args)
            profiler_config_dict = json.loads(args["profiler_config"])
            self.assertEqual(profiler_config_dict["torch_profiler_dir"], "/tmp/test/agent_loop_rollout_replica_0")
            self.assertTrue(profiler_config_dict["torch_profiler_with_stack"])
            self.assertTrue(profiler_config_dict["torch_profiler_record_shapes"])
            self.assertTrue(profiler_config_dict["torch_profiler_with_memory"])
            self.assertEqual(profiler_config_dict["delay_iterations"], 0)
            self.assertEqual(profiler_config_dict["max_iterations"], 0)

    def test_build_vllm_profiler_args_skips_legacy_env(self):
        """vLLM >= 0.13.0 rejects the VLLM_TORCH_PROFILER_* names, so they must stay unset."""
        tool_config = TorchProfilerToolConfig(contents=["stack"])
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)

        with patch.dict(os.environ, {}, clear=True):
            args = build_vllm_profiler_args(config, tool_config, rank=0, legacy_env=False)

            for name in (
                "VLLM_TORCH_PROFILER_DIR",
                "VLLM_TORCH_PROFILER_WITH_STACK",
                "VLLM_TORCH_PROFILER_RECORD_SHAPES",
                "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY",
            ):
                self.assertNotIn(name, os.environ)

            # The argument-based config is still built.
            profiler_config_dict = json.loads(args["profiler_config"])
            self.assertEqual(profiler_config_dict["torch_profiler_dir"], "/tmp/test/agent_loop_rollout_replica_0")

    def test_build_vllm_profiler_args_with_profile_window(self):
        tool_config = TorchProfilerToolConfig(contents=["stack"], profile_token_start=12, profile_token_end=46)
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)

        args = build_vllm_profiler_args(config, tool_config, rank=1)
        profiler_config_dict = json.loads(args["profiler_config"])
        self.assertEqual(profiler_config_dict["delay_iterations"], 12)
        self.assertEqual(profiler_config_dict["max_iterations"], 34)

    def test_build_vllm_profiler_args_with_npu_profile_window(self):
        tool_config = NPUToolConfig(contents=["npu"], profile_token_start=5, profile_token_end=13)
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        args = build_vllm_profiler_args(config, tool_config, rank=0)
        profiler_config_dict = json.loads(args["profiler_config"])
        self.assertEqual(profiler_config_dict["delay_iterations"], 5)
        self.assertEqual(profiler_config_dict["max_iterations"], 8)

    def test_rollout_trace_dir_matches_what_the_engines_write_to(self):
        """The finish hook derives its directory from this helper, so it must not drift."""
        tool_config = TorchProfilerToolConfig(contents=["cpu"])
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        expected = rollout_trace_dir(config, rank=2)

        self.assertEqual(build_sglang_profiler_args(config, tool_config, rank=2)["output_dir"], expected)
        with patch.dict(os.environ, {}, clear=True):
            args = build_vllm_profiler_args(config, tool_config, rank=2, legacy_env=False)
        self.assertEqual(json.loads(args["profiler_config"])["torch_profiler_dir"], expected)

    def test_build_sglang_profiler_args(self):
        # Case 1: Basic features
        tool_config = TorchProfilerToolConfig(contents=["stack", "shapes", "memory"])
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        with self.assertWarns(UserWarning):
            args = build_sglang_profiler_args(config, tool_config, rank=0)
        self.assertEqual(args["output_dir"], "/tmp/test/agent_loop_rollout_replica_0")
        self.assertTrue(args["with_stack"])
        self.assertTrue(args["record_shapes"])
        self.assertIsNone(args["start_step"])
        self.assertIsNone(args["num_steps"])

    def test_build_sglang_profiler_args_with_profile_window(self):
        tool_config = TorchProfilerToolConfig(contents=["stack"], profile_token_start=7, profile_token_end=16)
        config = ProfilerConfig(save_path="/tmp/test", tool_config=tool_config)
        args = build_sglang_profiler_args(config, tool_config, rank=0)
        self.assertEqual(args["start_step"], 7)
        self.assertEqual(args["num_steps"], 9)


class TestRolloutTraceRelocation(unittest.TestCase):
    """Engines write into a sub-directory; relocation brings the traces to the configured path."""

    def _config(self, save_path: str, **kwargs) -> ProfilerConfig:
        return ProfilerConfig(
            tool="torch",
            enable=True,
            ranks=[0],
            save_path=save_path,
            tool_config=TorchProfilerToolConfig(contents=["cpu"], discrete=True),
            **kwargs,
        )

    def _write_engine_traces(self, config: ProfilerConfig, rank: int, *names: str) -> str:
        src_dir = rollout_trace_dir(config, rank)
        os.makedirs(src_dir, exist_ok=True)
        for name in names:
            with open(os.path.join(src_dir, name), "w") as f:
                f.write(name)
        return src_dir

    def test_traces_end_up_next_to_the_training_ones(self):
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)
            src_dir = self._write_engine_traces(config, 2, "host_123.pt.trace.json.gz")

            relocated = relocate_rollout_traces(config, rank=2)

            expected = os.path.join(save_path, "rollout-replica2_host_123.pt.trace.json.gz")
            self.assertEqual(relocated, [expected])
            # The replica is in the name now that it is no longer in the path.
            self.assertTrue(os.path.exists(expected))
            self.assertEqual(os.listdir(src_dir), [])

    def test_traces_stay_where_the_engine_wrote_them_by_default(self):
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path)
            src_dir = self._write_engine_traces(config, 0, "host_123.pt.trace.json.gz")

            self.assertEqual(relocate_rollout_traces(config, rank=0), [])

            self.assertEqual(os.listdir(src_dir), ["host_123.pt.trace.json.gz"])

    def test_an_engine_that_wrote_nothing_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as save_path:
            config = self._config(save_path, relocate_results=True)

            self.assertEqual(relocate_rollout_traces(config, rank=1), [])


class TestRolloutFinishHook(unittest.TestCase):
    """The finish hook runs the user's command once, on the last profiled step (run_command=True)."""

    def _config(self, save_path: str, cmd: str, **kwargs) -> ProfilerConfig:
        return ProfilerConfig(
            tool="torch",
            enable=True,
            ranks=[0],
            save_path=save_path,
            tool_config=TorchProfilerToolConfig(contents=["cpu"], discrete=True),
            finish_hook_cmd=cmd,
            **kwargs,
        )

    def test_run_finish_hook_runs_command_without_going_through_stop(self):
        with tempfile.TemporaryDirectory() as save_path:
            marker = os.path.join(save_path, "hook_ran")
            profiler = DistProfiler(rank=0, config=self._config(save_path, f"touch {marker}"))

            profiler.run_finish_hook()

            self.assertTrue(os.path.exists(marker))

    def test_run_finish_hook_exports_save_path_to_the_command(self):
        with tempfile.TemporaryDirectory() as save_path:
            out = os.path.join(save_path, "env")
            profiler = DistProfiler(rank=0, config=self._config(save_path, f'echo "$VERL_PROFILE_SAVE_PATH" > {out}'))

            profiler.run_finish_hook()

            with open(out) as f:
                self.assertEqual(f.read().strip(), save_path)

    def test_run_finish_hook_skips_unselected_ranks(self):
        with tempfile.TemporaryDirectory() as save_path:
            marker = os.path.join(save_path, "hook_ran")
            config = self._config(save_path, f"touch {marker}", finish_hook_ranks=[3])
            profiler = DistProfiler(rank=0, config=config)

            profiler.run_finish_hook()

            self.assertFalse(os.path.exists(marker))

    def test_run_finish_hook_defers_the_command_until_the_final_step(self):
        # The command runs once, on the last profiled step: earlier steps pass run_command=False, so
        # nothing is uploaded yet (backend stop + relocation still happen elsewhere every step), and
        # the last step passes run_command=True to fire the single upload of the whole save_path.
        with tempfile.TemporaryDirectory() as save_path:
            marker = os.path.join(save_path, "hook_ran")
            profiler = DistProfiler(rank=0, config=self._config(save_path, f"touch {marker}"))

            profiler.run_finish_hook(run_command=False)
            self.assertFalse(os.path.exists(marker))  # not the last profiled step yet

            profiler.run_finish_hook(run_command=True)
            self.assertTrue(os.path.exists(marker))  # last step: command runs exactly once

    def test_run_finish_hook_uploads_the_whole_save_path_once(self):
        # A single end-of-run upload of the directory sends each accumulated trace exactly once --
        # this is what replaces the per-step upload (which re-sent earlier steps every step).
        with tempfile.TemporaryDirectory() as save_path:
            uploaded = os.path.join(save_path, "uploaded.log")
            cmd = f'for f in "$VERL_PROFILE_SAVE_PATH"/*.json.gz; do echo "$(basename "$f")" >> {uploaded}; done'
            profiler = DistProfiler(rank=0, config=self._config(save_path, cmd))

            # Both steps' traces have accumulated in save_path by the time the command runs.
            for name in ("step1.json.gz", "step2.json.gz"):
                with open(os.path.join(save_path, name), "w") as f:
                    f.write("x")

            profiler.run_finish_hook(run_command=True)

            sent = open(uploaded).read().split()
            self.assertEqual(sorted(sent), ["step1.json.gz", "step2.json.gz"])  # each exactly once


class TestBuildRolloutDistProfiler(unittest.TestCase):
    """Rollout `ranks` are global GPU ranks; the helper maps each to the replica that owns it."""

    def _config(self, **kwargs) -> ProfilerConfig:
        return ProfilerConfig(tool=None, enable=True, save_path="/tmp/test", tool_config=None, **kwargs)

    def _selected(self, replica_rank: int, world_size: int, **cfg) -> bool:
        profiler = build_rollout_dist_profiler(replica_rank, world_size, config=self._config(**cfg))
        return profiler.check_enable() and profiler.check_this_rank()

    def test_global_ranks_map_to_the_owning_replica(self):
        # world_size 8 (e.g. tp=8): global rank 0 lives on replica 0, global rank 8 lives on replica 1.
        self.assertTrue(self._selected(0, 8, ranks=[0, 8]))
        self.assertTrue(self._selected(1, 8, ranks=[0, 8]))
        # This is the whole point of the fix: [0, 8] no longer means replica indices 0 and 8.
        self.assertFalse(self._selected(8, 8, ranks=[0, 8]))
        self.assertFalse(self._selected(2, 8, ranks=[0, 8]))

    def test_a_rank_inside_a_replica_selects_that_whole_replica(self):
        # Any global rank owned by a replica selects it, not just the replica's base rank.
        self.assertTrue(self._selected(0, 8, ranks=[3]))  # rank 3 is on replica 0
        self.assertTrue(self._selected(1, 8, ranks=[9]))  # rank 9 is on replica 1
        self.assertFalse(self._selected(0, 8, ranks=[9]))

    def test_all_ranks_profiles_every_replica(self):
        for replica_rank in range(4):
            self.assertTrue(self._selected(replica_rank, 8, all_ranks=True, ranks=[0]))

    def test_empty_ranks_defaults_to_the_replica_owning_global_rank_0(self):
        self.assertTrue(self._selected(0, 8, ranks=[]))
        self.assertFalse(self._selected(1, 8, ranks=[]))

    def test_world_size_one_keeps_global_ranks_equal_to_replica_ranks(self):
        # tp=dp=pp=1: the replica index equals the global rank, so [0, 8] selects replicas 0 and 8.
        self.assertTrue(self._selected(0, 1, ranks=[0, 8]))
        self.assertTrue(self._selected(8, 1, ranks=[0, 8]))
        self.assertFalse(self._selected(1, 1, ranks=[0, 8]))

    def test_source_config_is_not_mutated(self):
        config = self._config(ranks=[0, 8])
        build_rollout_dist_profiler(1, 8, config=config)
        # The caller's (frozen) config still holds the original global ranks.
        self.assertEqual(list(config.ranks), [0, 8])


class TestServerProfilerFunctionality(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_start_stop_profile(self):
        try:
            # Import strictly inside test to avoid import errors if dependencies missing
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
        except ImportError:
            self.skipTest("vllm or dependencies not installed")
            return

        # Mock dependencies
        mock_profiler = MagicMock()
        mock_profiler.check_enable.return_value = True
        mock_profiler.check_this_rank.return_value = True
        mock_profiler.is_discrete_mode.return_value = True
        mock_profiler.config = ProfilerConfig(save_path="/tmp/test", tool_config=None)

        mock_engine = AsyncMock()

        # Mock self object
        mock_self = MagicMock()
        mock_self.node_rank = 0
        mock_self.replica_rank = 3
        mock_self.profiler_controller = mock_profiler
        mock_self.engine = mock_engine
        mock_self._should_profile = partial(vLLMHttpServer._should_profile, mock_self)

        # Test start_profile using the unbound method
        await vLLMHttpServer.start_profile(mock_self)
        mock_engine.start_profile.assert_called_once()

        # Test stop_profile
        with patch("verl.workers.rollout.vllm_rollout.vllm_async_server.relocate_rollout_traces") as mock_relocate:
            await vLLMHttpServer.stop_profile(mock_self)
        mock_engine.stop_profile.assert_called_once()
        # Relocation runs every profiled step so the engine's traces accumulate in save_path.
        mock_relocate.assert_called_once_with(mock_profiler.config, mock_self.replica_rank)
        # The engine does NOT run the finish command itself: it shares save_path with the colocated
        # training worker, whose single end-of-run upload covers these relocated traces too. Running
        # it here as well would upload the shared directory twice.
        mock_profiler.run_finish_hook.assert_not_called()

    async def test_vllm_stop_profile_skips_relocation_when_not_profiled(self):
        try:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
        except ImportError:
            self.skipTest("vllm or dependencies not installed")
            return

        mock_profiler = MagicMock()
        mock_profiler.check_enable.return_value = True
        mock_profiler.check_this_rank.return_value = False
        mock_profiler.is_discrete_mode.return_value = True

        mock_engine = AsyncMock()

        mock_self = MagicMock()
        mock_self.node_rank = 0
        mock_self.profiler_controller = mock_profiler
        mock_self.engine = mock_engine
        mock_self._should_profile = partial(vLLMHttpServer._should_profile, mock_self)

        with patch("verl.workers.rollout.vllm_rollout.vllm_async_server.relocate_rollout_traces") as mock_relocate:
            await vLLMHttpServer.stop_profile(mock_self)
        mock_engine.stop_profile.assert_not_called()
        mock_relocate.assert_not_called()

    async def test_vllm_start_stop_profile_non_master_node(self):
        try:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
        except ImportError:
            self.skipTest("vllm or dependencies not installed")
            return

        mock_profiler = MagicMock()
        mock_profiler.check_enable.return_value = True
        mock_profiler.check_this_rank.return_value = True
        mock_profiler.is_discrete_mode.return_value = True

        mock_engine = AsyncMock()

        mock_self = MagicMock()
        mock_self.node_rank = 1  # non-master node, should skip
        mock_self.profiler_controller = mock_profiler
        mock_self.engine = mock_engine
        mock_self._should_profile = partial(vLLMHttpServer._should_profile, mock_self)

        await vLLMHttpServer.start_profile(mock_self)
        mock_engine.start_profile.assert_not_called()

        await vLLMHttpServer.stop_profile(mock_self)
        mock_engine.stop_profile.assert_not_called()

    async def test_sglang_start_stop_profile(self):
        try:
            # Import strictly inside test to avoid import errors if dependencies missing
            from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer
        except ImportError:
            self.skipTest("sglang or dependencies not installed")
            return

        # Mock dependencies
        mock_profiler = MagicMock()
        mock_profiler.check_enable.return_value = True
        mock_profiler.check_this_rank.return_value = True
        mock_profiler.is_discrete_mode.return_value = True
        mock_profiler.config = ProfilerConfig(save_path="/tmp/test", tool_config=None)
        mock_profiler.tool_config = MagicMock()

        mock_tokenizer_manager = AsyncMock()

        mock_self = MagicMock()
        mock_self.profiler_controller = mock_profiler
        mock_self.tokenizer_manager = mock_tokenizer_manager
        mock_self.replica_rank = 0

        # Mock build_sglang_profiler_args to return known dict
        with patch("verl.workers.rollout.sglang_rollout.async_sglang_server.build_sglang_profiler_args") as mock_build:
            mock_args = {"arg1": "val1"}
            mock_build.return_value = mock_args

            # Test start_profile
            await SGLangHttpServer.start_profile(mock_self)

            mock_build.assert_called_once_with(mock_profiler.config, mock_profiler.tool_config, mock_self.replica_rank)
            mock_tokenizer_manager.start_profile.assert_called_once_with(**mock_args)

            # Test stop_profile
            with patch(
                "verl.workers.rollout.sglang_rollout.async_sglang_server.relocate_rollout_traces"
            ) as mock_relocate:
                await SGLangHttpServer.stop_profile(mock_self)
            mock_tokenizer_manager.stop_profile.assert_called_once()
            # Relocation runs every profiled step so traces accumulate in save_path; the engine does
            # not run the finish command itself (the colocated training worker's single end-of-run
            # upload covers them), so running it here too would upload the shared directory twice.
            mock_relocate.assert_called_once_with(mock_profiler.config, mock_self.replica_rank)
            mock_profiler.run_finish_hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
