# Academic writing guideline

Rules for every report and paper in this repository, including the living
teleoperation paper in `documents/paper/`. Follow them when writing new
sections and when editing old ones. The living-paper rule in `CLAUDE.md`
points here: any branch that touches the paper must also respect this
guideline.

## 1. Justify every design decision

For each method or experiment design decision, write down:

- **What** was implemented, in enough detail that a reader could
  reimplement it without reading our code.
- **Why** — the solid reason for the choice (a measurement, a derivation,
  a benchmark result, or a citation).

If there is no solid reason, say so explicitly. Use this convention in
LaTeX and Markdown alike:

> ⚠ *Unjustified choice:* value X was set by intuition / convention and
> has not been validated.

In LaTeX, use the `\unjustified{}` macro (defined in the paper preamble;
it renders as **[!]** *Unjustified choice:*) — pdflatex cannot typeset
the raw ⚠ character.

Do not dress a guess up as a derivation. A flagged guess is honest and
reviewable; a hidden one is a defect.

## 2. Flow diagram before LaTeX

Before writing or rewriting any section or subsection, produce a
**paragraph flow diagram**: one line per planned paragraph stating its
job and how it hands over to the next. Store these as Markdown files
under `documents/paper/<paper>/flow/` (one file per section, updated
whenever the section changes). Write the LaTeX only after the flow
diagram reads well on its own.

Example shape:

```
P1  hook: why the problem matters            -> narrows to our setting
P2  the specific obstacle in our setting     -> motivates the idea
P3  the idea in one paragraph                -> forward links to details
P4  contributions list                       -> maps to section structure
```

## 3. Language

- **British English** throughout (behaviour, colour, minimise,
  visualisation).
- **Active voice.** "We measure the error", not "the error is measured".
- **Simple and straightforward English.** No sentence longer than two
  printed lines. If a sentence needs a semicolon and two subclauses,
  split it.
- Define every acronym and named instrument at first use (e.g.
  NASA-TLX, SUS, DoF), in one clause, even if it is standard in the
  field.

## 4. Structure

- **Every section opens with an introductory paragraph** that says what
  the section does and forward-links to its subsections
  (`Section~\ref{...}`), so a reader always knows where they are and
  what is coming.
- **The abstract stays high level.** Write it for a general roboticist:
  what system, what problem, what kind of contribution. Do **not** put
  specific numbers, performance figures, or parameter values in the
  abstract — those live in the experiments section.

## 5. No code artefacts in prose

Do not put repository file paths, directory names, class names, or
function names in the paper. The paper describes the *system*, not the
*codebase*. Name components by what they do ("the method adapter",
"the rehearsal tool", "the benchmark runner"), and name methods by
their method name (`armplane`, `telegrip`, `dls`, ...), which is a
label, not a code path. Command-line snippets belong in the repository
README and the Markdown docs, not in the paper.

## 6. Numbers and claims

- Every quantitative claim in the body must be reproducible from a
  table, a figure, or a stated derivation.
- When benchmark conditions change, regenerate the affected tables and
  update every hard-coded number quoted in the prose in the same
  branch (living-paper rule).
- State the conditions a number was measured under (scene, rates,
  parameter set) or point to the section that does.
