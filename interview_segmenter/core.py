"""Data model, loaders and the answer-window rule.

A Questionnaire is an ordered list of Items; a Transcript is a list of Units
(lines / sentences). Aligners answer in unit indices; times are attached
afterwards if the units have them.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Item:
    """One thing the interviewer is scripted to say (a question, or a battery sub-item)."""
    item_id: str
    text: str
    categories: dict[str, str] | None = None   # code -> label, for coding closed answers


@dataclass
class Unit:
    """Smallest piece of transcript an item can be mapped onto."""
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str = ""


@dataclass
class Transcript:
    units: list[Unit]

    @property
    def timed(self) -> bool:
        return all(u.start is not None for u in self.units)


@dataclass
class Match:
    """Item -> units [from_unit, to_unit)."""
    item_id: str
    from_unit: int
    to_unit: int          # exclusive
    score: float | None = None
    start: float | None = None
    end: float | None = None


# --- loaders --------------------------------------------------------------------

def load_items(path: str | Path) -> list[Item]:
    """Questionnaire in order. .json: [{"id", "text", "categories"?: {code: label}}, ...]; .csv: columns id,text;
    anything else: one item per line (id = line number)."""
    p = Path(path)
    if p.suffix == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix == ".csv":
        with p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [{"id": str(k), "text": t} for k, t in enumerate(p.read_text(encoding="utf-8").splitlines()) if t.strip()]
    return [Item(str(r["id"]), r["text"].strip(), r.get("categories")) for r in rows]


def load_transcript(path: str | Path) -> Transcript:
    """.json: [{"text", "start"?, "end"?, "speaker"?}, ...] (or {"units": [...]});
    anything else: one utterance per line, optionally "Speaker: text"."""
    p = Path(path)
    if p.suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw = raw["units"] if isinstance(raw, dict) else raw
        return Transcript([Unit(u["text"], u.get("start"), u.get("end"), u.get("speaker", "")) for u in raw])
    units = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        head, sep, rest = line.partition(":")
        is_label = sep and 0 < len(head) <= 20 and " " not in head.strip()
        units.append(Unit(rest.strip(), speaker=head.strip()) if is_label else Unit(line.strip()))
    return Transcript(units)


def split_speakers(tr: Transcript, interviewer: str) -> tuple[Transcript, Transcript]:
    """One interleaved transcript -> (interviewer, respondent). Untimed units get their
    position as time, so the answer rule works on turn order alone."""
    units = [Unit(u.text, k if u.start is None else u.start, k if u.end is None else u.end, u.speaker)
             for k, u in enumerate(tr.units)]
    return (Transcript([u for u in units if u.speaker == interviewer]),
            Transcript([u for u in units if u.speaker != interviewer]))


def split_words(words: list[dict], how: str = "gap", gap_s: float = 0.6) -> Transcript:
    """ASR words [{"text", "start", "end"}] -> units. how: "gap" (new unit after a pause
    > gap_s), "sentence" (after . ? !) or "sat" (SaT sentence splitter, needs wtpsplit);
    combine with "+", e.g. "sat+gap"."""
    words = [w for w in words if w["text"].strip()]
    starts = {0}
    for h in how.split("+"):
        if h == "gap":
            starts |= {i for i in range(1, len(words)) if words[i]["start"] - words[i - 1]["end"] > gap_s}
        elif h == "sentence":
            starts |= {i for i in range(1, len(words)) if words[i - 1]["text"].strip().endswith((".", "?", "!"))}
        elif h == "sat":
            starts |= sat_starts([w["text"].strip() for w in words])
        else:
            raise ValueError(h)
    cuts = sorted(starts) + [len(words)]
    return Transcript([Unit(" ".join(w["text"].strip() for w in words[a:b]), words[a]["start"], words[b - 1]["end"])
                       for a, b in zip(cuts, cuts[1:])])


_sat = None


def sat_starts(tokens: list[str], model: str = "sat-12l-sm") -> set[int]:
    """Indices of tokens that start a sentence according to SaT (robust to unpunctuated ASR output)."""
    global _sat
    if _sat is None:
        from wtpsplit import SaT
        _sat = SaT(model)
    ends, pos = [], 0
    for s in _sat.split(" ".join(tokens), strip_whitespace=False):
        pos += len(s)
        ends.append(pos)
    starts, pos = set(), 0
    for i, t in enumerate(tokens):
        if i and any(pos - 1 <= e <= pos for e in ends):
            starts.add(i)
        pos += len(t) + 1
    return starts


# --- times and answers -------------------------------------------------------------

def attach_times(matches: list[Match], tr: Transcript) -> list[Match]:
    """Fill start/end from the matched units (no-op for untimed transcripts)."""
    for m in matches:
        span = [u for u in tr.units[m.from_unit:m.to_unit] if u.start is not None]
        if span:
            m.start, m.end = span[0].start, span[-1].end
    return matches


def answer_spans(asked: list[Match]) -> dict[str, tuple[float, float]]:
    """Time span of each item's answer: from the start of its first occurrence to the start
    of the next item asked. The question's end is deliberately not used: respondents often
    answer while options are still being read. `asked` must be timed (attach_times);
    untimed matches are ignored."""
    timed = sorted((m for m in asked if m.start is not None), key=lambda m: m.start)
    out = {}
    for i, m in enumerate(timed):
        if m.item_id not in out:
            out[m.item_id] = (m.start, next((x.start for x in timed[i + 1:] if x.item_id != m.item_id), float("inf")))
    return out


def answer_windows(asked: list[Match], respondent: Transcript) -> dict[str, tuple[int, int]]:
    """Respondent units [from, to) that start inside each item's answer span. Both
    transcripts must be on one clock. Items without speech in their span get no window."""
    out = {}
    for item_id, (lo, hi) in answer_spans(asked).items():
        idx = [k for k, u in enumerate(respondent.units) if u.start is not None and lo <= u.start < hi]
        if idx:
            out[item_id] = (idx[0], idx[-1] + 1)
    return out


def answer_dialogues(asked: list[Match], interviewer: Transcript, respondent: Transcript,
                     labels: tuple[str, str] = ("I", "R")) -> dict[str, str]:
    """The answer span with both speakers, one "I: ..." / "R: ..." line per unit in time
    order. Interviewers often echo or categorise the answer ("so, rather satisfied?")."""
    out = {}
    for item_id, (lo, hi) in answer_spans(asked).items():
        lines = sorted((u.start, lab, u.text) for tr, lab in zip((interviewer, respondent), labels)
                       for u in tr.units if u.start is not None and lo <= u.start < hi)
        out[item_id] = "\n".join(f"{lab}: {t}" for _, lab, t in lines)
    return out
