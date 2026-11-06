"""Shared helpers for the MES project."""

import json
import re


def parse_json_response(text, default=None):
    """
    Parse a JSON object or array from an LLM response.

    Handles markdown code fences, leading/trailing prose and responses that
    are already Python objects.  Returns ``default`` on failure.
    """
    if isinstance(text, (dict, list)):
        return text
    if text is None:
        return default
    text = str(text).strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for open_char, close_char in (("{", "}"), ("[", "]")):
            start, end = candidate.find(open_char), candidate.rfind(close_char)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start:end + 1])
                except json.JSONDecodeError:
                    continue
    return default
