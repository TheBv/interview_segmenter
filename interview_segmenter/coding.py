"""Coding answers into a closed question's answer categories with an LLM, and flagging
answers whose code differs from the value entered during the interview.

The reply is one sentence of reasoning, then "Code: <code>". The server enforces this
with a regex (vLLM `structured_outputs`), so any instruct model works unchanged. Asking
for the reason first matters: without it, models code "two" as the category labelled 6
more often.
"""

import json
import re
import urllib.request

from .align import LLM_MODEL, LLM_URL

SYSTEM = ("You code answers from a survey interview (CAPI-style, read aloud by an interviewer). "
          "Given the survey question, its answer categories and what was said, decide which category the "
          "respondent chose. Use only what the respondent says; an interviewer's echo counts only if the "
          "respondent confirms it or does not object. If no answer to this question can be recognised, "
          "choose \"none\". Respondents often answer with the number of a category (\"two\" = category 2). "
          "Reply with one short sentence of reasoning, then a line 'Code: <code>'.")


def prompt(question: str, categories: dict[str, str], said: str, dialogue: bool = False) -> str:
    cats = "\n".join(f"{c}: {l}" for c, l in categories.items())
    who = "Interview excerpt (I = interviewer, R = respondent)" if dialogue else "Respondent's answer"
    return (f"Question:\n{question}\n\nAnswer categories:\n{cats}\n\n{who}:\n{said or '(nothing)'}\n\n"
            "Which category did the respondent choose?")


def parse(reply: str) -> tuple[str, str]:
    """'<reason>\\nCode: <code>' -> (code, reason); code "error" if the reply has no code line."""
    m = re.search(r"Code: (\S+)\s*$", reply)
    return (m.group(1) if m else "error"), reply.split("\n")[0]


def code_answer(question: str, categories: dict[str, str], said: str, dialogue: bool = False,
                url: str = LLM_URL, model: str = LLM_MODEL, thinking: bool = False) -> tuple[str, str]:
    """One answer, one request -> (category code or "none", reason). `said` is the
    respondent's answer window, or with dialogue=True the window with both speakers
    (core.answer_dialogues)."""
    codes = "|".join(re.escape(str(c)) for c in [*categories, "none"])
    body = {"model": model, "temperature": 0, "max_tokens": 6000 if thinking else 200,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "structured_outputs": {"regex": rf"[^\n]{{1,300}}\nCode: ({codes})"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt(question, categories, said, dialogue)}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return parse(json.loads(r.read())["choices"][0]["message"]["content"])


def flag(codes: dict[str, str], entered: dict[str, str]) -> list[str]:
    """Items whose coded answer differs from the entered value: the ones worth listening to."""
    return [k for k, v in entered.items() if k in codes and codes[k] != str(v)]
