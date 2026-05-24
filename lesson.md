# Lessons

- Symptom: paper diagnostic names drifted from code-facing trace keys and the lightweight figure CLI could not build its parser. Cause: the velocity-variation proxy was exposed under older terminology and the figure module was not exercised at parser level. Fix: use velocity-variation difficulty keys end to end, keep local-defect PTG as the primary diagnostic, and test the figure parser/payload path. Prevention: add formula, payload, and stale-token tests when renaming paper-facing diagnostics.
