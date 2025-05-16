---
title: DATE-LM Leaderboard
emoji: 🏆
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# DATE-LM Data Attribution Leaderboards

This repo contains the leaderboard code associated with the DATE-LM Paper. The leaderboards
are hosted in [this HuggingFace Space](https://huggingface.co/spaces/DataAttributionEval/DATE-LM-Leaderboard).

The leaderboards are split into 2 broad categories: Training Data Selection and Applications.
Each category contains 3 leaderboards, as indicated below.

- Pre-Training (10K)
- Pre-Training (30K)
- Fine-Tuning

| Category                           | Leaderboards                                        |
| ---------------------------------- | --------------------------------------------------- |
| Training Data Selection            | Pre-Training (10K), Pre-Training (30K), Fine-Tuning |
| Applications (Toxicity / Bias)     | Homogeneous, Heterogeneous                          |
| Applications (Factual Attribution) | Factual Attribution                                 |

Details on the tasks corresponding to each leaderboard as well as their code pipelines
can be found in the DATE-LM paper and [Github repo](https://github.com/DataAttributionEval/DATE-LM).

## Submission

To submit to the leaderboard: submit via the form in the "Submit Scores" tab on the HuggingFace Space page. This will open up a pull request in this repo. It will need to be merged by a member of the team in order to be displayed in the HuggingFace Space.

Information for Submission include:

- Influence Scores File
- Paper/Code/Contact Link
- Method Name and Category
- Metrics (dependent on leaderboard chosen)
- and more

## Ranking

Each leaderboard's ranking is based on the values from the metrics, with details specified in the description of each leaderboard. To summarize, the leaderboards are ranked using the following schemes:

| Leaderboard                            | Ranking Metric                                     |
| -------------------------------------- | -------------------------------------------------- |
| Pre-Training (10K), Pre-Training (30K) | highest score in **avg** column                    |
| Fine-Tuning                            | average of **MMLU**, **GSM8K**, and **BBH** scores |
| Applications (Toxicity / Bias)         | highest score in **AUPRC** column                  |
| Applications (Factual Attribution)     | average of **Recall@50** and **MRR** scores        |

## Repo Files

Overview of Repo files:

- app file: `app.py`
  - mappings files: `filePaths.py`, `nameMapping.py`
  - Github PR creation file: `pr.py`
- submissions storage: `submissions` folder
  - Note: each submission has its own dedicated folder containing `metadata.json` and the influence scores
- leaderboards data: `data` folder
- github workflow files:
  - Add submission into leaderboard json upon merge: `merge-data.yml`, `scripts/merge_data.py`
  - Sync repo with HuggingFace Space: `push-to-hf.yml`
