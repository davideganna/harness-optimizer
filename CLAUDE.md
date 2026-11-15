# Project instructions

## Keep MODULES.md in sync (mandatory)

`MODULES.md` holds a short summary of every module so the user can read it instead of the code.
Whenever you add, remove, rename, or change the behavior or public API of any module under
`src/harness_optimizer/` (or the examples or test layout), update the matching entry in
`MODULES.md` in the same change, and update its "Last updated" date. A code change is not done
until `MODULES.md` matches it.

Style: plain, non-technical language that describes what each file contains and is for. Avoid
function signatures, parameter lists and implementation details; mention a code name only when
the reader needs it.
