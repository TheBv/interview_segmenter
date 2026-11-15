"""Model-free checks of the solver and the answer rule: python test_segmenter.py"""

from interview_segmenter import Item, Match, Transcript, Unit, answer_windows, attach_times, solve, split_speakers

items = [Item("q1", "a"), Item("q2", "b"), Item("q3", "c")]

# global matching keeps questionnaire order: q2's best candidate (unit 0) comes before q1's,
# so the solver takes q2's weaker candidate after q1; q3 has nothing above the floor -> not asked
cands = {"q1": [Match("q1", 1, 2, score=0.9)],
         "q2": [Match("q2", 0, 1, score=0.95), Match("q2", 3, 4, score=0.7)],
         "q3": [Match("q3", 5, 6, score=0.5)]}
got = [(m.item_id, m.from_unit) for m in solve(items, cands, floor=0.6)]
assert got == [("q1", 1), ("q2", 3)], got

# answer windows on one interleaved, untimed transcript (turn order as time)
tr = Transcript([Unit("Frage eins?", speaker="I"), Unit("Ja.", speaker="R"),
                 Unit("Frage zwei?", speaker="I"), Unit("Hm, zwei.", speaker="R"), Unit("Genau.", speaker="R")])
interviewer, respondent = split_speakers(tr, "I")
asked = attach_times([Match("q1", 0, 1), Match("q2", 1, 2)], interviewer)
assert answer_windows(asked, respondent) == {"q1": (0, 1), "q2": (1, 3)}
print("ok")
