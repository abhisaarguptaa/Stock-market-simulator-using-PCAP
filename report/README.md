# Report

| File | What it is |
|---|---|
| [`Report_IEEE.pdf`](Report_IEEE.pdf) | Final report, IEEE conference template |
| [`Report_Springer.pdf`](Report_Springer.pdf) | Same report, Springer Nature (LNCS) conference template |

Both versions have the same content and the cover page from the project synopsis. Sections, in the
order the mini-project guidelines ask for: Title, Abstract, Keywords, Introduction, Literature
Review, Problem Statement, Objectives, Proposed Methodology, System Architecture, Algorithms,
Experimental Setup, Results, Performance Analysis, Comparison with Existing Methods, Conclusion,
Future Scope, References (IEEE format).

## Sources

- `main-ieee.tex`, `main-springer.tex`: the two templates. They differ only in the title block.
- `content/`: the text, shared by both (`cover`, `abstract`, `litreview`, `body1` to `body3`).
- `refs.bib`: references. `figures/`: plots made from `results/` and terminal screenshots.

## Rebuild the PDFs

```
sudo apt install -y texlive-latex-extra texlive-science texlive-publishers
bash report/build.sh
```

## Where the screenshots come from

- `exec_determinism_colab.png`, `exec_colab_env.png`: text output of the Colab notebook on a Tesla T4.
- `exec_demo.png`, `exec_tests.png`: `scripts/demo.sh` and `lob_tests` on a 4-core Linux machine.
