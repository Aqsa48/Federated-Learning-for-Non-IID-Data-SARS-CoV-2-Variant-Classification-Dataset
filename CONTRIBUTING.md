# Contributing

Start with a small change that someone can run or verify. Report the problem, show expected and actual behavior, and include a minimal public or synthetic example. Do not include credentials, personal data, or restricted research inputs.

## Local checks

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python baseline.py --demo
```

Use a virtual environment with Python 3.10 or later. Explain the checks you ran in your pull request. For new behavior, add a focused test with an explicit expected result. Keep unrelated changes in a separate pull request.

## Small tasks

- Document verified feature-generation steps and train-only fitting boundaries.
- Add a temporal or geographic evaluation policy with an explicit synthetic test.
- Prepare a dataset card with verified provenance and source acknowledgements.

Discuss larger changes in an issue before implementing them. Respect contributors, describe disagreements in terms of the code or evidence, and credit sources and contributors. Maintainer responses depend on availability; no response time is promised.
