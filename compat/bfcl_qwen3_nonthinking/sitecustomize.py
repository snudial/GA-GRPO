"""Opt-in non-thinking prefix for the pinned Qwen BFCL completion handler."""
import os
import sys


def install():
    if os.environ.get("BFCL_QWEN3_NONTHINKING") != "1":
        return
    # The vLLM server consumes already formatted completion prompts. Only the
    # BFCL client needs this handler patch (and has its parser dependencies).
    if sys.prefix == os.environ.get("BFCL_VLLM_VENV"):
        return
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    original = QwenFCHandler._format_prompt
    if getattr(original, "_jump_nonthinking", False):
        return

    def format_prompt(self, messages, function):
        prompt = original(self, messages, function)
        if not prompt.endswith("<|im_start|>assistant\n"):
            raise ValueError("Unexpected Qwen FC generation prompt suffix")
        return prompt + "<think>\n\n</think>\n\n"

    format_prompt._jump_nonthinking = True
    QwenFCHandler._format_prompt = format_prompt


install()
