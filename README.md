---
title: QASports Website - Basketball
emoji: 👁
colorFrom: purple
colorTo: green
sdk: streamlit
sdk_version: 1.33.0
python_version: 3.9
suggested_hardware: t4-small
app_file: app.py
pinned: false
license: mit
tags:
  - sports
  - question-answering
  - open-domain-qa
  - extractive-qa
short_description: "QASports the first large sports-themed QA dataset"
models:
  - deepset/roberta-base-squad2
datasets:
  - PedroCJardim/QASports
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Website

This website presents a collection of documents from the dataset named "QASports", the first large sports question answering dataset for open questions. QASports contains real data of players, teams and matches from the sports soccer, basketball and American football.

- 💻 Website: https://huggingface.co/spaces/leomaurodesenv/qasports-website
- 🔧 Scripts: https://github.com/leomaurodesenv/qasports-dataset-website

> **Note**. As first version, we are only focusing in Basketball data.

## Dataset Summary

QASports is the first large sports-themed question answering dataset counting over 1.5 million questions and answers about 54k preprocessed wiki pages, using as documents the wiki of 3 of the most popular sports in the world, Soccer, American Football and Basketball. Each sport can be downloaded individually as a subset, with the train, test and validation splits, or all 3 can be downloaded together.

- 🎲 Dataset: https://huggingface.co/datasets/PedroCJardim/QASports
- 🔧 Scripts: https://github.com/leomaurodesenv/qasports-dataset-scripts/
