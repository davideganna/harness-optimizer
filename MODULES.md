# Module Guide

A plain-language tour of the project: what each file contains and what it is for. Read this
instead of the code to see how the pieces fit together. **The file is kept in sync with the code:
when a module changes, its description here changes too.**

Last updated: 2026-11-16

## The big picture

The project improves prompts automatically, and measures carefully whether each change really
helped. It has two halves:

1. **A measuring instrument (the evaluation harness).** It takes a prompt and a set of test
   examples, runs a language model on every example, scores the answers, and reports how good
   the prompt is, including how confident we can be in that number.
2. **An optimizer built on top of it.** Small AI "agents" look at the examples the prompt gets
   wrong, guess why, and propose edits to the prompt. Each edited prompt is measured with the
   harness, and it is kept only if it is clearly better, not better by luck.

Everything talks to language models through one shared doorway (the gateway). By default the
models run for free on your own computer through Ollama.

The source code lives in `src/harness_optimizer/`, organized in folders:

| Folder | In one sentence |
|---|---|
| `core/` | The basic building blocks: prompts, examples, datasets, results. |
| `llm/` | Everything about talking to language models. |
| `eval/` | The measuring instrument: running prompts, scoring answers, statistics. |
| `agents/` | The AI helpers that diagnose failures and propose prompt edits, plus the loop that drives them. |
| `memory/` | The archive of every prompt version and every measurement. |
| `policy/` | The spending and stopping limits. |
| `search/` | Deciding which new prompts deserve more measuring, and whether one replaces the current best. |
| `obs/` | The logbook that records everything that happens. |
| `experiments/` | Running the optimizer many times to find out which of its parts really help. |

---

## Top-level files

### `__init__.py`: front door of the package
Lists the most commonly used pieces so they can be imported straight from the package, and
records the version number (0.1.0, still in development).

### `config.py`: loading secret keys
Reads a private `.env` file, which is where you would keep an API key for Claude. It never
overwrites settings you already have.

### `cli.py`: the command-line tool
What runs when you type `harness-opt` in a terminal. It has four commands:
- **eval**: measures one prompt on a dataset, saves the results in the `runs/` folder, and
  prints the score and a few of the worst mistakes.
- **optimize**: starts the automatic optimizer on a prompt. It prints what happens in each
  round, then saves the best prompt it found and the history of the edits that led to it.
- **compare**: takes two saved runs and says whether one prompt is really better than the other.
- **ablate**: runs several versions of the optimizer (for example with and without the Critic)
  several times each, with different random splits of the data, and prints a table comparing
  how good their final prompts were and how many model calls they used.

Options control which model to use, how many rounds to run, how much data goes to training
and how much to validation, and so on. Switches turn the optional parts of the optimizer on or
off (the cheaper "successive halving" screening, the Critic, the lessons between rounds), so
you can measure whether each part actually helps. Its log messages and its own printed output
go to the same place, so they appear in the right order. The optimizer's agents use a slightly larger local model
(`qwen3:4b`) than the default model being tested (`llama3.2:3b`).

---

## `core/`: the basic building blocks

### `prompt.py`: what a prompt is
A prompt is stored as a list of named **sections** rather than one block of text. Examples
are "system" (who the model should be), "task" (what to do) and "output_format" (how to
answer). Splitting it up lets the optimizer make precise edits ("rewrite the output format")
and lets us see later which parts of a prompt changed. Prompts contain blanks like `{{text}}`
that are filled in with each example's data. This file also handles loading and saving prompts
as YAML files, filling in the blanks, and giving each prompt a unique "fingerprint" so
duplicates are easy to spot.

### `candidate.py`: prompt versions and their family tree
Every prompt version the optimizer considers is a **candidate**. Each candidate remembers its
parent and what edit produced it (which section, why, and which suspected problem it was
fixing). Together the candidates form a family tree of prompts, which shows the history of what
was tried and what worked.

### `dataset.py`: examples and datasets
An **example** is an input plus the expected answer (for instance a review and its sentiment).
A **dataset** is a collection of examples loaded from a file. This file also splits a dataset
into three groups, always the same way for a given random seed:
- **train**: the agents may look at these mistakes to learn from them.
- **validation**: used to decide whether a new prompt is better. The agents only ever see the
  overall score, never the individual examples.
- **test**: kept completely hidden until the very end, for one honest final measurement.

### `results.py`: what a measurement produces
Holds the outcome of an evaluation: the score of each example (always between 0 and 1), what
the model answered, any errors, how long it took and how much it cost. The overall **report**
can compute the average score with a confidence range, list the failures (worst first) in a
readable form for the agents, and be saved to or loaded from disk.

### `task.py`: which model to use
Describes a model setting: which provider (Ollama, Claude, or a fake one for testing), which
model, and how long answers may be. It also stores the default choices: `llama3.2:3b` as the
model being tested, and `qwen3:4b` for the agents, which need more reasoning ability. In a live
test the smaller model never managed to make a working edit, while `qwen3:4b` fixed the prompt
in one round.

---

## `llm/`: talking to language models

### `base.py`: the common vocabulary
Defines what a request to a model and a response from it look like, regardless of provider.
It also defines the two kinds of errors: temporary ones worth retrying (overload, network
hiccups) and permanent ones (bad request, missing model).

### `gateway.py`: the single doorway
Every model call in the project goes through here, which keeps the following in one place:
- **Reusing answers**: identical requests are served from a saved copy.
- **Retrying**: temporary errors are retried, with growing pauses between attempts.
- **Pacing**: it caps how many calls run at the same time.
- **Bookkeeping**: it tracks calls, tokens and cost separately for each role (the model being
  tested, the judge, each agent).
- **Logging**: every call is written to the logbook.

### `cache.py`: the answer archive
Stores model answers so that re-running an experiment costs nothing and gives exactly the same
results. It offers three variants: no storage, storage in memory, or a small database file on
disk.

### `ollama_client.py`: the local model connection (default)
Connects to Ollama, the free program that runs open models on your own computer. It is tuned
for a small graphics card: it skips the hidden "thinking" step and keeps the memory window
modest. It turns Ollama's errors into clear messages, for example "the model isn't downloaded,
run ollama pull" or "Ollama doesn't seem to be running", including the known fix for older
NVIDIA drivers.

### `anthropic_client.py`: the Claude connection (optional)
Connects to Anthropic's Claude models when you want them; this needs an API key. On purpose,
it never silently switches to another model when Claude refuses a request, because that would
distort the measurement.

### `mock.py`: a fake model for tests
Pretends to be a model and answers through a simple rule you give it. It is used for tests and
offline demos, and it can be told to fail a few times on purpose to check the retry logic.

### `pricing.py`: the price list
The cost per token of each Claude model, used to report how much a run cost. Local Ollama models
and the fake model are free.

---

## `eval/`: the measuring instrument

### `runner.py`: running a prompt on a dataset
Takes one prompt version and one dataset, fills the prompt in for every example, sends all of
them to the model (in parallel), scores each answer, and produces a report. Two important habits:
it checks first that the prompt can be filled in for every example, so a broken prompt fails
before anything is spent, and an answer that failed counts as a score of 0 instead of being
ignored, so error-prone prompts don't look better than they are.

### `stats.py`: is the difference real?
The statistics behind confident conclusions. It computes a confidence range for a score, and
compares two prompts **example by example** on the same data to estimate how likely it is that
the new one is really better rather than lucky. It uses a resampling technique called the
bootstrap.

### `metrics/`: ways of scoring an answer
- **`base.py`**: the common shape every scoring method follows, including the option to pull the
  actual answer out of a longer reply before scoring.
- **`deterministic.py`**: simple, rule-based scorers that give the same result every time:
  exact match (ignoring case and stray punctuation), "contains the expected text", pattern
  matching, "is valid JSON (optionally of a required shape)", and "the final number is correct".
  It also offers a way to turn any custom function into a scorer.
- **`composite.py`**: combines several scorers into one. Some scorers can act as must-pass gates,
  for example "the answer must be valid JSON, then grade its content".
- **`judge.py`**: uses a language model as a grader. It gives the model a rubric, the input, the
  reference answer and the answer to grade, then turns the grade into a 0–1 score. It records
  the grader's reasoning so a person can check it. Making these judges more trustworthy is on the
  roadmap.
- **`__init__.py`**: lists the scorers available by name from the command line.

---

## `agents/`: the AI helpers and the optimization loop

### `base.py`: getting reliable structured answers from agents
Agents have to reply in a precise structured format (JSON). This file asks for that format,
checks the reply, and, if something is wrong, tells the model what was wrong and asks again a
limited number of times. Besides format errors it can catch problems in the content (for
example, a prompt edit that loses a `{{blank}}`). With Ollama it also forces the model to stay
within the required format as it writes. It also contains helpers to load the agents'
instructions and to show a prompt to an agent section by section, marking which sections
contain blanks that must be kept.

### `analyst.py`: the Failure Analyst
Reads the current prompt and the training examples it got wrong, then proposes a few
**hypotheses**: likely reasons for the failures, which examples each one explains, and what
could be changed in the prompt. Example ids the model made up are discarded. It is also shown
the lessons learned in earlier rounds.

### `proposer.py`: the Proposer
Turns one hypothesis into one concrete edit of the prompt: rewrite a section, add a new section,
or remove one. It checks that the edit makes sense: the section exists, the content isn't empty,
and no blank like `{{text}}` is lost. If something is wrong it explains the problem to the model
and lets it try again (up to three times) instead of discarding the idea. This was added after
the first live run, where the small model dropped the `{{text}}` blank in every single edit.
Like the Analyst, it is shown the lessons from earlier rounds.

### `critic.py`: the Critic (optional)
Reviews each proposed edit **before** any money or time is spent measuring it. It sees the
section before and after the edit and the problem the edit is meant to fix, and rejects only
clear mistakes: answers to specific examples written into the prompt, contradictions with
another section, a changed task, or instructions too vague to matter. When unsure it approves,
because the measurement is the final judge. A rejection must say what is wrong. It costs one
agent call per edit, so it is off by default; whether it pays for itself is a question for an
ablation.

### `reflector.py`: the Reflector
After each round, reads what happened (which ideas were tried, which edits were rejected and
why, how each one scored, which was accepted) and writes one or two short, general **lessons**,
such as "rewriting the task section to demand a single word fixed the format problem". The
lessons are saved and shown to the Analyst and the Proposer in later rounds, so they can build
on what worked and avoid repeating what failed. It sees only overall validation scores, never
validation examples. Lessons it already wrote are not repeated.

### `guards.py`: a quick safety check
A free, instant check before spending any effort measuring a new prompt. It rejects prompts that
lost or invented blanks, exact duplicates of prompts already tried, prompts that are far too
long, and edits that paste a failed training example's text word for word (the most obvious way
of "cheating" by writing an answer for one example into the prompt).

### `pipeline.py`: the optimization loop
Runs the whole process, round after round:
1. Measure the current best prompt on a small batch of training examples and collect its
   mistakes.
2. The Analyst explains the mistakes.
3. The Proposer writes several candidate edits.
4. The safety check removes broken or duplicate candidates, and the Critic (if switched on)
   rejects clearly bad edits.
5. Each surviving candidate is tried on training examples, and the ones that beat the current
   best are kept. This happens in one of two ways: everyone on the same small batch, or
   "successive halving" (see `search/halving.py`).
6. The most promising ones are measured on the validation set. If one is **clearly** better
   than the current best, it becomes the new best.
7. The Reflector (on by default) writes lessons for the next rounds.

The loop stops when the budget runs out, when several rounds pass without progress, or when
there is nothing left to fix. At the very end, the original prompt and the best prompt are each
measured once on the hidden test set, for an honest final comparison. Each round is reported
as it happens, and everything is saved in the archive.

### `prompts/`: the agents' own instructions
Text files holding the instructions given to each agent (Analyst, Proposer, Critic, Reflector).
They tell the agents to look for causes that a prompt change can fix, to keep every blank, to
prefer adding a new section over rewriting one, to write general rules rather than answers to
specific examples, to reject only clear mistakes, and to keep lessons short and reusable.

---

## `memory/`: the archive

### `store.py`
Keeps every prompt version, every measurement and every lesson made during an optimization run.
It can trace any prompt back through its ancestors to the original, hand out the most recent
lessons, and save and load the whole archive on disk.

### `lessons.py`
Describes a lesson (the round it came from and its text) and formats a list of lessons for an
agent to read.

---

## `policy/`: limits

### `budget.py`
The rules for when to stop: a maximum number of rounds, model calls, money or time, and a
"patience" limit (stop after so many rounds without improvement). These limits are enforced by
the code itself; the agents are never just asked to respect them.

---

## `search/`: deciding what wins

### `acceptance.py`
The rule for replacing the current best prompt: the challenger must score higher on the
validation set, and the statistics must say the gain is unlikely to be luck. This keeps the
optimizer from "improving" on random noise.

### `halving.py`: successive halving
A cheaper way to try many candidates. All of them are first measured on a handful of training
examples; only the better half moves on to a bigger handful, and so on. Each bigger handful
contains the smaller one, so earlier answers are reused for free. The current best prompt is
measured on every step too, so comparisons are always fair. Only candidates that beat it on the
largest step count as promising. Planned: other search strategies (beam, evolutionary).

---

## `obs/`: the logbook

### `tracing.py`
Writes a timestamped record of everything that happens (each model call, each error, each
round, each time an agent was asked to fix its answer) to a log file, one line per event, so a
run can be inspected afterwards. Planned: readable run reports and replaying past runs.

---

## `experiments/`: research runs

### `ablation.py`: which parts of the optimizer help?
An **ablation** switches parts of a system off one at a time to see what each part contributes.
This file defines ready-made variants of the optimizer: `base` (the plain loop), `reflector`
(plus lessons), `critic` (plus the Critic), `halving` (plus successive halving) and `full`
(everything). It runs every variant once per random seed; the seed changes how the data is split,
so the spread between seeds shows how much of a difference is real rather than luck. All variants
of the same seed use exactly the same data split, so they are compared fairly.

For each run it records the validation and test scores before and after, how many rounds and
improvements there were, how long it took, and how many model calls were made, separately for
the model being tested and for the agents. Answers served from the saved-answer archive still
count as calls, so a variant doesn't look cheaper just because it ran after another one.
Results are written to disk after every run. If a long ablation is interrupted, running it again
skips the runs that already finished, and runs that crashed are retried. A crash in one run is
recorded instead of stopping the whole ablation. At the end it writes a summary table.

---

## Outside the package

- **`examples/dates/`**: a harder benchmark. The task is to find the date in a short text and
  write it as `YYYY-MM-DD`. The data follows rules the weak starting prompt never mentions:
  answer with the date only, numbers like `04/03/2025` are day-first (4 March), two-digit years
  mean 20xx, and dates hide in sentences with times and amounts. The optimizer has to discover
  these rules from mistakes, over several rounds. A script generates the 120 examples (the same
  ones every time), each tagged with its date style so splits stay balanced. A hand-written
  reference prompt stating every rule checks that the task is doable: the weak prompt scores
  0.00 and the reference prompt 0.97 with the default local model.
- **`examples/sentiment/`**: a small worked example that classifies reviews as positive,
  negative or neutral. It contains the labeled reviews, a deliberately weak prompt (it never
  mentions "neutral"), an improved prompt, a fake model that reacts to the wording of the prompt
  so improvements show up even offline, and a demo script that compares the two prompts.
- **`tests/`**: automated tests that check each part of the project. There is one file per area
  (prompts, datasets, gateway, scorers, runner, statistics, agents, pipeline, successive
  halving, ablations, limits and archive, command line, and the two model connections).
