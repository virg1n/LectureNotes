"""LectureNotes: a harness-guided small-LM agent for LaTeX lecture notes.

The package is organised around one idea: the language model never produces a
whole document in a single shot. An orchestrator decomposes the job into small,
verifiable tasks (plan -> section -> figure -> repair -> cheat sheet). Every
artifact passes through deterministic verifiers (LaTeX lint + compile, sandboxed
figure execution + figure lint, length / coverage checks) whose structured
feedback drives targeted repairs. The very same verifiers are reused as
rewards for RL, as filters for teacher-distilled training data and as the
evaluation benchmark.
"""

__version__ = "0.1.0"

# Bump whenever the wording/format of task prompts changes: training data built
# with one prompt version should be served with the same version.
PROMPT_VERSION = "v1"
