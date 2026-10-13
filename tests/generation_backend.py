"""CPU integration-test generation backend using the real tokenizer and model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from bfcl_env.rollout_protocol import Generation, PolicyVersion

__all__ = [
    "TransformersGenerationBackend", "ThinkingTemplateRefused", "render_chat_prompt"
]


class ThinkingTemplateRefused(RuntimeError):
    """The chat template opens a reasoning block the environment cannot parse."""


_THINK_MARKERS = ("<think>", "<thinking>", "<|thinking|>")


def render_chat_prompt(
    tokenizer: Any,
    conversation: Sequence[Mapping[str, str]],
    *,
    allow_thinking: bool = False,
    empty_tool_call_assistant_history: bool = False,
    check_thinking: bool = True,
) -> str:
    """The one prompt renderer shared by local Transformers and remote vLLM."""
    messages = [dict(message) for message in conversation]
    if empty_tool_call_assistant_history:
        for message in messages:
            if (
                message.get("role") == "assistant"
                and "<tool_call>" in str(message.get("content", ""))
            ):
                message["content"] = ""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if check_thinking and not allow_thinking:
        lowered = text.lower()
        for marker in _THINK_MARKERS:
            if marker in lowered:
                raise ThinkingTemplateRefused(
                    f"the chat template emits {marker!r}; the rollout would be "
                    "reasoning text rather than an action the environment can "
                    "execute. Use a non-thinking checkpoint or template."
                )
    return text


@dataclass
class TransformersGenerationBackend:
    """Sample the next action from a local causal LM."""

    model: Any
    tokenizer: Any
    policy_step: int = 0
    model_checksum: str | None = None
    default_max_new_tokens: int = 256
    top_k: int | None = None
    """The BFCL baseline omits top-k; training ablations pass it explicitly."""

    top_p: float | None = None
    allow_thinking: bool = False
    empty_tool_call_assistant_history: bool = False
    """Reproduce the legacy BFCL v1.3 QwenFCHandler history bug.

    Its parser stores calls in a structured ``tool_calls`` field, but its manual
    prompt formatter renders only ``content``.  Consequently a previous tool-call
    assistant message is an empty assistant turn in the next prompt. The
    author-lineage default is False because its handler fixed this behavior.
    """
    _checked: bool = field(default=False, init=False, repr=False)

    def policy_version(self) -> PolicyVersion:
        return PolicyVersion(
            policy_step=self.policy_step, model_checksum=self.model_checksum
        )

    # ---- template ---------------------------------------------------------- #

    def _render(self, conversation: Sequence[Mapping[str, str]]) -> str:
        text = render_chat_prompt(
            self.tokenizer, conversation,
            allow_thinking=self.allow_thinking,
            empty_tool_call_assistant_history=self.empty_tool_call_assistant_history,
            check_thinking=not self._checked,
        )
        self._checked = True
        return text

    # ---- generation -------------------------------------------------------- #

    def generate(
        self,
        prompts: Sequence[Sequence[Mapping[str, str]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[Generation]:
        import torch  # noqa: PLC0415

        if not prompts:
            return []

        texts = [self._render(conversation) for conversation in prompts]
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, padding_side="left"
        ).to(device)

        was_training = self.model.training
        use_cache = getattr(self.model.config, "use_cache", None)
        cpu_state = torch.random.get_rng_state() if seed is not None else None
        cuda_state = (
            torch.cuda.get_rng_state_all()
            if seed is not None and torch.cuda.is_available()
            else None
        )
        self.model.eval()
        self.model.config.use_cache = True
        try:
            if seed is not None:
                torch.manual_seed(int(seed))
            generation_kwargs = {
                "max_new_tokens": max_new_tokens or self.default_max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else None,
                "pad_token_id": (
                    self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                ),
            }
            if self.top_k is not None:
                generation_kwargs["top_k"] = self.top_k
            if self.top_p is not None:
                generation_kwargs["top_p"] = self.top_p
            with torch.no_grad():
                output = self.model.generate(
                    **encoded,
                    **generation_kwargs,
                )
        finally:
            if use_cache is not None:
                self.model.config.use_cache = use_cache
            if was_training:
                self.model.train()
            if cpu_state is not None:
                torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

        prompt_length = encoded["input_ids"].shape[1]
        version = self.policy_version()
        generations: list[Generation] = []
        for row in output:
            token_ids = [
                int(t) for t in row[prompt_length:]
                if int(t) != (self.tokenizer.pad_token_id or -1)
            ]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            finish_reason = "stop"
            if stop:
                # Cut at the first stop string and say so, rather than handing the
                # environment a trailing second action it never asked for.
                cuts = [text.find(s) for s in stop if text.find(s) >= 0]
                if cuts:
                    text = text[: min(cuts)]
                    finish_reason = "stop_string"
            elif len(token_ids) >= (max_new_tokens or self.default_max_new_tokens):
                finish_reason = "length"
            generations.append(
                Generation(
                    text=text.strip(),
                    token_ids=token_ids,
                    finish_reason=finish_reason,
                    policy_version=version,
                )
            )
        return generations
