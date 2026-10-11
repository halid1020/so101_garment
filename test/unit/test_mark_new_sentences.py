"""Marking the sentences a report gained since an older version.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_mark_new_sentences
"""

import unittest

from tool.mark_new_sentences import mark, pool, sentences, strip_marks


def chunks(text):
    return [text[s:e] for s, e, _ in sentences(text)]


class TestSplitting(unittest.TestCase):
    def test_abbreviations_and_decimals_do_not_end_a_sentence(self):
        text = "We use e.g. ACT and Fig. 3 shows 0.5 s. Then it stops."
        self.assertEqual(
            chunks(text), ["We use e.g. ACT and Fig. 3 shows 0.5 s.", "Then it stops."]
        )

    def test_a_sentence_may_start_with_pi05(self):
        self.assertEqual(
            chunks("It is done. pi0.5 is slow."), ["It is done.", "pi0.5 is slow."]
        )

    def test_blank_lines_items_and_labels_are_boundaries(self):
        text = "\\begin{itemize}\n\\item One thing\n\\item Another\n\\end{itemize}\n\nNext para."
        self.assertEqual(chunks(text), ["One thing", "Another", "Next para."])

    def test_a_float_contributes_only_its_caption(self):
        text = (
            "\\begin{table}[t]\n\\centering\n\\caption{Errors per model. Lower is"
            " better.}\n\\label{tab:x}\n\\input{tables/x}\n\\end{table}"
        )
        self.assertEqual(chunks(text), ["Errors per model.", "Lower is better."])

    def test_display_maths_and_comments_are_skipped(self):
        text = "Before it.\n\\[ a = b. C \\]\n% A comment. Here\nAfter it."
        self.assertEqual(chunks(text), ["Before it.", "After it."])


class TestMarking(unittest.TestCase):
    OLD = "ACT plans 100 steps. Touch helps a little."

    def test_an_old_sentence_stays_and_a_changed_number_is_marked(self):
        text = "ACT plans 100 steps. Touch helps a lot. ACT plans 50 steps."
        out, n = mark(text, pool([self.OLD]))
        self.assertEqual(n, 2)
        self.assertEqual(
            out,
            "ACT plans 100 steps. \\bl{Touch helps a lot.} \\bl{ACT plans 50 steps.}",
        )

    def test_colour_macros_and_braces_do_not_make_a_sentence_new(self):
        out, n = mark("\\gn{ACT plans \\new{100} steps.}", pool([self.OLD]))
        self.assertEqual(n, 0)

    def test_a_sentence_crossing_a_brace_is_wrapped_in_balanced_pieces(self):
        text = "\\gn{\\textbf{New idea.} It works well. Touch helps a little.}"
        out, n = mark(text, pool([self.OLD]))
        self.assertEqual(n, 2)
        self.assertEqual(
            out,
            "\\gn{\\bl{\\textbf{New idea.}} \\bl{It works well.} Touch helps a little.}",
        )
        self.assertEqual(out.count("{"), out.count("}"))

    def test_a_sentence_opening_a_group_it_does_not_close(self):
        text = "Start here \\emph{and go on. Then} stop."
        out, _ = mark(text, set())
        self.assertEqual(out.count("{"), out.count("}"))
        self.assertIn("\\emph{\\bl{and go on.}", out)

    def test_a_new_heading_is_marked_inside_its_braces(self):
        out, _ = mark("\\section{Future work}\n\\label{sec:f}\nText.", set())
        self.assertTrue(out.startswith("\\section{\\bl{Future work}}"))

    def test_stripping_is_the_identity_and_marking_is_idempotent(self):
        text = "\\gn{\\textbf{New idea.} It works.}\n\nTouch helps a little. More $x.$ Here."
        once, _ = mark(text, pool([self.OLD]))
        self.assertEqual(strip_marks(once), text)
        self.assertEqual(mark(once, pool([self.OLD]))[0], once)


if __name__ == "__main__":
    unittest.main()
