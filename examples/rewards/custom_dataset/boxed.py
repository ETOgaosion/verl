# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.utils.dataset.rl_dataset import RLHFDataset


class BoxedRLHFDataset(RLHFDataset):
    def _build_messages(self, example: dict, key: str):
        messages = super()._build_messages(example, key)

        last_user_msg = messages[-1]
        assert last_user_msg["role"] == "user"
        last_user_msg["content"] += (
            "\n\n"
            "Let's think step by step to solve the problem and "
            'provide the final answer in the format of "\\boxed{...}".'
        )

        return messages
