# interview_segmenter

Finds where an interviewer reads each questionnaire item in a transcript, and which respondent speech answers it.

```bash
pip install -e .
python -m interview_segmenter items.json --interviewer interviewer.json --respondent respondent.json
python -m interview_segmenter items.txt --transcript interview.txt --interviewer-label I
python -m interview_segmenter items.json --interviewer I.json --respondent R.json --code dialogue --entered entered.json
```

- **Items**: `.json` (`[{"id": ..., "text": ..., "categories": {"1": "yes", "2": "no"}}]`, categories optional), `.csv` (`id,text`) or one item per line, in questionnaire order.
- **Transcripts**: `.json` (`[{"text", "start", "end", "speaker"}]`, times optional) or one utterance per line (`Speaker: text`). Two separate transcripts must share one clock.
- **Methods**: `--method embedding` (default, sentence embeddings) or `--method llm` (any OpenAI-compatible server with JSON-schema output, e.g. vLLM; set `--llm-url` / `--llm-model`).

- **Coding**: `--code respondent|dialogue` codes each answer of an item with categories into one category (or `none`) with an LLM, from the respondent's speech or from the window with both speakers. `--entered` (`{"item": code}`) lists the items whose code differs from the entered value. Needs a vLLM server (regex-constrained output).

Output is JSON with the question segments, the items not asked, the answer segments (respondent text and the dialogue with both speakers), and, if requested, codes and flagged items.

Test: `python test_segmenter.py`
