"""Read a completed metric dictionary without relying on process log newlines."""
import ast
import io
import math
import re
import tokenize


def completed_eval_metrics(text):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    markers = list(re.finditer(r"Eval-only completed:\s*", text))
    if not markers:
        raise ValueError("No completed evaluation marker")
    payload = text[markers[-1].end():]
    if not payload.startswith("{"):
        raise ValueError("Completed evaluation has no metric dictionary")
    depth = 0
    lines = payload.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(payload).readline):
        if token.type != tokenize.OP:
            continue
        if token.string == "{":
            depth += 1
        elif token.string == "}":
            depth -= 1
            if depth == 0:
                row, column = token.end
                end = sum(len(line) for line in lines[:row - 1]) + column
                metrics = ast.literal_eval(payload[:end])
                score = metrics.get("eval_default_pass1")
                count = metrics.get("eval_num_samples")
                if (type(score) not in (int, float) or not math.isfinite(score)
                        or not 0 <= score <= 1 or type(count) not in (int, float)
                        or not math.isfinite(count) or count <= 0 or int(count) != count):
                    raise ValueError("Invalid evaluation score or sample count")
                return metrics
    raise ValueError("Incomplete evaluation metric dictionary")
