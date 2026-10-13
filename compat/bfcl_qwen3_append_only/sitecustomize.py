"""Use the exported append-only tokenizer for BFCL prompt formatting."""
import os
import sys


def install():
    if os.environ.get("BFCL_QWEN3_APPEND_ONLY") != "1":
        return
    # Only the evaluator client imports BFCL; training and serving use other venvs.
    if sys.prefix != os.environ.get("BFCL_RUNTIME_VENV"):
        return
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    def format_prompt(self, messages, function):
        if self.tokenizer.init_kwargs.get("jump_grpo_serialization") != "qwen3_append_only_v1":
            raise ValueError("Append-only evaluation requires an append-only checkpoint tokenizer")
        return self.tokenizer.apply_chat_template(
            messages, tools=function, tokenize=False, add_generation_prompt=True,
        )

    QwenFCHandler._format_prompt = format_prompt


install()
