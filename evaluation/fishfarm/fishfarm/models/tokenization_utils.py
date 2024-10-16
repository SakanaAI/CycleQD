import dataclasses
from typing import Optional

from transformers import PreTrainedTokenizerBase

from .base import Message


class MaskedTokens:

    text: str
    token_ids: list[int]
    mask: list[bool]

    def __init__(self) -> None:
        self.text = ""
        self.token_ids = []
        self.mask = []

    def extend(
        self,
        messages: list[Message],
        mask_value: bool,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: Optional[str],
        add_generation_prompt: bool,
    ) -> None:
        if len(messages) == 0:
            # `tokenizer.apply_chat_template` does not accept an empty list.
            raise ValueError("At least one message is required.")

        all_text: str = tokenizer.apply_chat_template(
            conversation=[dataclasses.asdict(message) for message in messages],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert all_text.startswith(self.text)
        new_text = all_text[len(self.text) :]
        new_token_ids: list[int] = tokenizer.encode(new_text, add_special_tokens=False)

        self.token_ids.extend(new_token_ids)
        self.mask.extend([mask_value] * len(new_token_ids))
        self.text = all_text


def tokenize_messages(
    messages: list[Message],
    tokenizer: PreTrainedTokenizerBase,
    chat_template: Optional[str],
) -> MaskedTokens:
    masked_tokens = MaskedTokens()

    for i, message in enumerate(messages):
        if message.role != "assistant":
            continue

        masked_tokens.extend(messages[:i], False, tokenizer, chat_template, True)
        masked_tokens.extend(messages[: i + 1], True, tokenizer, chat_template, False)

    masked_tokens.extend(messages, False, tokenizer, chat_template, True)
    return masked_tokens
