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
