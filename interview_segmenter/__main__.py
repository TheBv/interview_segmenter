"""python -m interview_segmenter ITEMS --interviewer I.json [--respondent R.json]
   python -m interview_segmenter ITEMS --transcript both.txt --interviewer-label I

Prints JSON: the located question segments and, if a respondent side is given,
the answer segment of each item (respondent only, and with the interviewer's lines).
--code codes the answers of items that have "categories" with an LLM; --entered
flags the items whose code differs from the entered value."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from .align import ALIGNERS, EMBEDDING_MODEL, LLM_MODEL, LLM_URL
from .coding import code_answer, flag
from .core import answer_dialogues, answer_windows, attach_times, load_items, load_transcript, split_speakers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", help="questionnaire in order (.json, .csv or one item per line)")
    ap.add_argument("--interviewer", help="interviewer transcript")
    ap.add_argument("--respondent", help="respondent transcript (same clock as the interviewer's)")
    ap.add_argument("--transcript", help="one transcript with both speakers (needs --interviewer-label)")
    ap.add_argument("--interviewer-label", help="speaker label of the interviewer in --transcript")
    ap.add_argument("--method", choices=ALIGNERS, default="embedding")
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    ap.add_argument("--floor", type=float, default=0.6, help="minimum similarity of a match (embedding)")
    ap.add_argument("--llm-url", default=LLM_URL)
    ap.add_argument("--llm-model", default=LLM_MODEL)
    ap.add_argument("--code", choices=("respondent", "dialogue"),
                    help="code closed answers from the respondent's window or the window with both speakers")
    ap.add_argument("--entered", help='entered responses, {"item": code} (.json), to flag disagreements')
    a = ap.parse_args()

    items = load_items(a.items)
    if a.transcript:
        if not a.interviewer_label:
            ap.error("--transcript needs --interviewer-label")
        interviewer, respondent = split_speakers(load_transcript(a.transcript), a.interviewer_label)
    elif a.interviewer:
        interviewer = load_transcript(a.interviewer)
        respondent = load_transcript(a.respondent) if a.respondent else None
    else:
        ap.error("give --interviewer or --transcript")

    kw = ({"model": a.embedding_model, "floor": a.floor} if a.method == "embedding"
          else {"url": a.llm_url, "model": a.llm_model})
    matches = attach_times(ALIGNERS[a.method](items, interviewer, **kw), interviewer)
    text = lambda tr, f, t: " ".join(u.text for u in tr.units[f:t])
    out = {"questions": [{"item": m.item_id, "from": m.from_unit, "to": m.to_unit, "start": m.start,
                          "end": m.end, "score": m.score, "text": text(interviewer, m.from_unit, m.to_unit)}
                         for m in matches],
           "not_asked": [it.item_id for it in items if it.item_id not in {m.item_id for m in matches}]}
    if respondent is not None:
        win, dia = answer_windows(matches, respondent), answer_dialogues(matches, interviewer, respondent)
        answers = [{"item": k, "from": win.get(k, (None,))[0], "to": win.get(k, (None, None))[1],
                    "text": text(respondent, *win[k]) if k in win else "", "dialogue": d} for k, d in dia.items()]
        if a.code:
            by_id = {it.item_id: it for it in items}
            todo = [x for x in answers if by_id[x["item"]].categories]
            ask = lambda x: code_answer(by_id[x["item"]].text, by_id[x["item"]].categories,
                                        x["dialogue" if a.code == "dialogue" else "text"],
                                        a.code == "dialogue", a.llm_url, a.llm_model)
            with ThreadPoolExecutor(16) as pool:
                for x, (c, r) in zip(todo, pool.map(ask, todo)):
                    x["code"], x["reason"] = c, r
            if a.entered:
                with open(a.entered, encoding="utf-8") as f:
                    out["flagged"] = flag({x["item"]: x["code"] for x in todo}, json.load(f))
        out["answers"] = answers
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
