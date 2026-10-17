# NewsBot Intelligence System

An end-to-end natural language processing portfolio project that classifies BBC news articles and explains what is happening in the text through sentiment, entity, part-of-speech, dependency, and TF-IDF analysis.

## Results

- **1,440** unique labeled articles analyzed after removing 50 exact duplicates
- **5** editorial categories: business, entertainment, politics, sport, and tech
- **96.9%** held-out accuracy and **96.7%** macro F1 with a Linear SVM
- Integrated outputs for category, sentiment, named entities, POS patterns, and syntax relationships

## Project structure

```text
ITAI2373-NewsBot-Midterm/
├── NewsBot_Intelligence_System.ipynb
├── newsbot_pipeline.py
├── README.md
├── requirements.txt
├── data/
├── figures/
├── models/
├── results/
└── report/
```

## Quick start

1. Create a Python environment and install `requirements.txt`.
2. Install the spaCy model with `python -m spacy download en_core_web_sm`.
3. Run `python newsbot_pipeline.py` to reproduce all outputs.
4. Open `NewsBot_Intelligence_System.ipynb` for the guided analysis.

The project is compatible with Google Colab's free tier. The notebook includes setup instructions.

## Dataset

This project uses the recommended [BBC News Classification competition dataset](https://www.kaggle.com/competitions/learn-ai-bbc/data). The included training file has 1,490 rows with article IDs, full article text, and one of five category labels. Competition terms apply to the data.

## Method

The system removes missing records and exact duplicate articles, normalizes common web noise, and uses a stratified 80/20 split with random seed 42. Three models are compared on the same TF-IDF representation: multinomial Naive Bayes, logistic regression, and Linear SVM. Linguistic analysis uses spaCy; article-level tone uses VADER.

## Responsible use

The model is a decision-support prototype, not an autonomous editorial authority. BBC-specific vocabulary and time period may not generalize to other publishers. Sentiment reflects lexical tone and may misinterpret quotation, irony, or domain-specific phrasing. Production use should include human review and ongoing drift monitoring.

## Author

Kameron Walters  
Individual project for ITAI 2373 Natural Language Processing
