"""Diagnostic JSON encoding only: never coerce actual API arguments or rewards."""
import json
import math
from pathlib import Path

from fastapi.encoders import jsonable_encoder


def diagnostic_json(value, seen=None, depth=0):
    """Keep normal FastAPI encoding; tag Python objects JSON cannot represent.

    Tags are for inspection, not executable replay. Do not invoke arbitrary
    __repr__ on unsupported model-created objects just to write a log.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'__python_type__': 'float', 'value': str(value)}
    seen = set() if seen is None else seen
    if depth > 64 or id(value) in seen:
        return {'__python_type__': type(value).__name__, '__log_note__': 'cycle_or_depth_limit'}
    if value is Ellipsis:
        return {'__python_type__': 'ellipsis', 'value': '...'}
    if value is NotImplemented:
        return {'__python_type__': 'NotImplementedType'}
    seen = seen | {id(value)}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: diagnostic_json(v, seen, depth+1) for k,v in value.items()}
        return {'__python_type__': 'dict', 'items': [
            [diagnostic_json(k, seen, depth+1), diagnostic_json(v, seen, depth+1)]
            for k,v in value.items()]}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [diagnostic_json(v, seen, depth+1) for v in value]
    try:
        encoded = jsonable_encoder(value)
        json.dumps(encoded, ensure_ascii=True, allow_nan=False)
        return encoded
    except Exception:
        return {'__python_type__': type(value).__name__, '__log_note__': 'not_json_serializable'}


def save_request_log(self, file_path):
    # Encode a detached representation. self.requests and REPL objects stay intact.
    records = [diagnostic_json(request) for request in self.requests]
    with Path(file_path).open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + '\n')
