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

"""The V1 trainer must drive the rollout engines' profiler, not just the worker groups."""

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _trainer(cls, **managers):
    trainer = cls.__new__(cls)
    trainer.config = OmegaConf.create({"global_profiler": {"profile_continuous_steps": False, "steps": [1]}})
    trainer.global_steps = 1
    trainer.prev_step_profile = False
    trainer.curr_step_profile = True
    trainer.next_step_profile = False
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.actor_rollout_wg = MagicMock()
    for name, manager in managers.items():
        setattr(trainer, name, manager)
    return trainer


def test_profiled_step_starts_and_stops_rollout_engines():
    manager = MagicMock()
    trainer = _trainer(_StubTrainer, llm_server_manager=manager)

    trainer._start_profiling()
    manager.start_profile.assert_called_once()

    trainer._stop_profiling()
    manager.stop_profile.assert_called_once()


def test_unprofiled_step_leaves_rollout_engines_alone():
    manager = MagicMock()
    trainer = _trainer(_StubTrainer, llm_server_manager=manager)
    trainer.curr_step_profile = False

    trainer._start_profiling()
    trainer._stop_profiling()

    manager.start_profile.assert_not_called()
    manager.stop_profile.assert_not_called()


def test_continuous_steps_leave_the_open_collection_running():
    # With profile_continuous_steps the collection opened by the previous step stays open, so a
    # continuing step must not restart it. Step boundaries are no longer marked at the trainer
    # level: the profiler is advanced once per update mini-batch inside the workers instead.
    manager = MagicMock()
    trainer = _trainer(_StubTrainer, llm_server_manager=manager)
    trainer.config = OmegaConf.create({"global_profiler": {"profile_continuous_steps": True, "steps": [1, 2]}})
    trainer.prev_step_profile = True
    trainer.curr_step_profile = True

    trainer._start_profiling()

    trainer.actor_rollout_wg.step_profile.assert_not_called()
    trainer.actor_rollout_wg.start_profile.assert_not_called()
    manager.start_profile.assert_not_called()


def test_separate_async_also_profiles_standalone_replicas():
    hybrid, standalone = MagicMock(), MagicMock()
    trainer = _trainer(
        PPOTrainerSeparateAsync,
        llm_server_manager=hybrid,
        standalone_server_manager=standalone,
    )

    trainer._start_profiling()
    trainer._stop_profiling()

    for manager in (hybrid, standalone):
        manager.start_profile.assert_called_once()
        manager.stop_profile.assert_called_once()
