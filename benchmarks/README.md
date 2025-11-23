Benchmarks directory
--------------------

This folder contains an auto-generated safe benchmark runner:

- agent_benchmark.py : safe runner. Edit DRY_RUN=False to actually run network calls.
- example_prompts.yaml : prompts used by the runner.

CAUTION: The generated runner defaults to DRY_RUN=True to avoid accidental live calls.
Review agent endpoints and credentials before executing.
