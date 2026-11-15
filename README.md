# interview_segmenter

Finds where an interviewer reads each questionnaire item in a transcript, and which respondent speech answers it.

```bash
pip install -e .
python -m interview_segmenter items.json --interviewer interviewer.json --respondent respondent.json
python -m interview_segmenter items.txt --transcript interview.txt --interviewer-label I
```

- **Items**: `.json` (`[{"id": ..., "text": ...}]`), `.csv` (`id,text`) or one item per line, in questionnaire order.
- **Transcripts**: `.json` (`[{"text", "start", "end", "speaker"}]`, times optional) or one utterance per line (`Speaker: text`). Two separate transcripts must share one clock.
- **Methods**: `--method embedding` (default, sentence embeddings) or `--method llm` (any OpenAI-compatible server with JSON-schema output, e.g. vLLM; set `--llm-url` / `--llm-model`).

Output is JSON with the question segments, the items not asked, and the answer segments.

Test: `python test_segmenter.py`
