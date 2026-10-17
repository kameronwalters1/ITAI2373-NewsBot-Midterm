from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "BBC News Train.csv"
PROJECT = ROOT
DATA = PROJECT / "data"
FIGURES = PROJECT / "figures"
RESULTS = PROJECT / "results"
MODELS = PROJECT / "models"

SEED = 42
PALETTE = ["#17324D", "#1F6F8B", "#2F9C95", "#E0A458", "#D95D39"]


def clean_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^A-Za-z0-9£$€%\-'.!? ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ensure_dirs() -> None:
    for path in (DATA, FIGURES, RESULTS, MODELS):
        path.mkdir(parents=True, exist_ok=True)


def save_plot(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()


def top_terms(vectorizer, model, labels, n=10):
    names = np.asarray(vectorizer.get_feature_names_out())
    rows = []
    if hasattr(model, "coef_"):
        for i, label in enumerate(labels):
            for rank, idx in enumerate(np.argsort(model.coef_[i])[-n:][::-1], 1):
                rows.append({"category": label, "rank": rank, "term": names[idx], "weight": float(model.coef_[i, idx])})
    return pd.DataFrame(rows)


def run_pipeline(project_dir: str | Path | None = None) -> dict:
    """Reproduce the complete analysis. project_dir is accepted for notebook convenience."""
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(SOURCE).rename(columns={"Text": "content", "Category": "category"})
    raw_rows = len(df)
    missing = int(df[["content", "category"]].isna().any(axis=1).sum())
    duplicates = int(df.duplicated(subset=["content"]).sum())
    df = df.dropna(subset=["content", "category"]).drop_duplicates(subset=["content"]).copy()
    df["clean_content"] = df["content"].map(clean_text)
    df["word_count"] = df["clean_content"].str.split().str.len()
    shutil.copy2(SOURCE, DATA / "BBC News Train.csv")

    counts = df["category"].value_counts().sort_index()
    ax = counts.plot(kind="bar", color=PALETTE, figsize=(8, 4.6), rot=0)
    ax.set(title="BBC News Articles by Category", xlabel="Category", ylabel="Articles")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom", fontsize=9)
    save_plot("01_category_distribution.png")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["category"], test_size=0.20, random_state=SEED, stratify=df["category"]
    )
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20000, sublinear_tf=True)
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.35),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=4.0, random_state=SEED),
        "Linear SVM": LinearSVC(C=1.0, random_state=SEED),
    }
    model_rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("tfidf", clone(vectorizer)), ("classifier", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        model_rows.append({"model": name, "accuracy": accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro")})
        fitted[name] = (pipe, pred)
    metrics = pd.DataFrame(model_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    metrics.to_csv(RESULTS / "model_comparison.csv", index=False)
    ax = metrics.set_index("model")[["accuracy", "macro_f1"]].plot(kind="bar", figsize=(8.2, 4.7), color=["#1F6F8B", "#E0A458"], rot=0)
    ax.set(title="Classification Performance on Held-Out Test Set", xlabel="", ylabel="Score", ylim=(0.80, 1.00))
    ax.legend(frameon=False, loc="lower right")
    save_plot("02_model_comparison.png")

    best_name = metrics.iloc[0]["model"]
    best_pipe, best_pred = fitted[best_name]
    labels = sorted(df["category"].unique())
    report = classification_report(y_test, best_pred, labels=labels, output_dict=True)
    pd.DataFrame(report).T.to_csv(RESULTS / "classification_report.csv")
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    plt.figure(figsize=(7.2, 5.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"Confusion Matrix for {best_name}")
    plt.xlabel("Predicted category"); plt.ylabel("Actual category")
    save_plot("03_confusion_matrix.png")
    joblib.dump(best_pipe, MODELS / "newsbot_classifier.joblib")

    terms = top_terms(best_pipe.named_steps["tfidf"], best_pipe.named_steps["classifier"], best_pipe.named_steps["classifier"].classes_)
    terms.to_csv(RESULTS / "top_tfidf_terms.csv", index=False)
    pivot = terms.pivot(index="rank", columns="category", values="term")
    pivot.to_csv(RESULTS / "top_terms_by_category.csv")
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.5), sharex=False)
    for ax, (cat, sub) in zip(axes, terms.groupby("category")):
        sub = sub.sort_values("weight")
        ax.barh(sub["term"], sub["weight"], color=PALETTE[labels.index(cat)])
        ax.set_title(cat.title()); ax.set_xlabel("Model weight"); ax.grid(axis="y", visible=False)
    fig.suptitle("Most Discriminative TF-IDF Features by Category", y=1.02, fontsize=14)
    save_plot("04_top_terms.png")

    nltk.download("vader_lexicon", download_dir=str(ROOT / "work" / "nltk_data"), quiet=True)
    nltk.data.path.insert(0, str(ROOT / "work" / "nltk_data"))
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["clean_content"].map(lambda x: sia.polarity_scores(x)["compound"])
    df["sentiment_label"] = pd.cut(df["sentiment"], [-1.01, -0.05, 0.05, 1.01], labels=["negative", "neutral", "positive"])
    sent = df.groupby("category", observed=True)["sentiment"].agg(["mean", "median", "std"]).reset_index()
    sent.to_csv(RESULTS / "sentiment_by_category.csv", index=False)
    plt.figure(figsize=(8, 4.8))
    sns.boxplot(data=df, x="category", y="sentiment", order=labels, palette=PALETTE, hue="category", legend=False)
    plt.title("VADER Sentiment Distribution by Category"); plt.xlabel("Category"); plt.ylabel("Compound sentiment")
    save_plot("05_sentiment_by_category.png")

    nlp = spacy.load("en_core_web_sm", disable=["textcat"])
    sample = df.groupby("category", group_keys=False).sample(n=min(100, counts.min()), random_state=SEED).copy()
    entity_counts = defaultdict(Counter); pos_counts = defaultdict(Counter); dep_counts = defaultdict(Counter)
    keep_entities = {"PERSON", "ORG", "GPE", "DATE", "MONEY"}
    for cat, doc in zip(sample["category"], nlp.pipe(sample["clean_content"].tolist(), batch_size=32)):
        entity_counts[cat].update(ent.label_ for ent in doc.ents if ent.label_ in keep_entities)
        pos_counts[cat].update(tok.pos_ for tok in doc if not tok.is_space and not tok.is_punct)
        dep_counts[cat].update(tok.dep_ for tok in doc if not tok.is_space and not tok.is_punct)
    ent_df = pd.DataFrame(entity_counts).fillna(0).astype(int).T.reindex(labels).fillna(0).astype(int)
    ent_df.to_csv(RESULTS / "entity_counts_by_category.csv")
    ent_df.plot(kind="bar", figsize=(9, 5), color=["#17324D", "#1F6F8B", "#2F9C95", "#E0A458", "#D95D39"], rot=0)
    plt.title("Named Entities in a Stratified 500-Article Sample"); plt.xlabel("Category"); plt.ylabel("Entity mentions"); plt.legend(title="Entity", frameon=False)
    save_plot("06_entities_by_category.png")

    pos_df = pd.DataFrame(pos_counts).fillna(0).T.reindex(labels).fillna(0)
    core_pos = [x for x in ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"] if x in pos_df.columns]
    pos_share = pos_df[core_pos].div(pos_df.sum(axis=1), axis=0) * 100
    pos_share.to_csv(RESULTS / "pos_distribution.csv")
    pos_share.plot(kind="bar", figsize=(9, 5), rot=0, color=sns.color_palette("crest", n_colors=len(core_pos)))
    plt.title("Part-of-Speech Distribution by Category"); plt.xlabel("Category"); plt.ylabel("Share of all tokens (%)"); plt.legend(frameon=False, ncol=3)
    save_plot("07_pos_distribution.png")

    dep_df = pd.DataFrame(dep_counts).fillna(0).T.reindex(labels).fillna(0)
    core_dep = [x for x in ["nsubj", "dobj", "pobj", "ROOT", "amod", "compound"] if x in dep_df.columns]
    dep_share = dep_df[core_dep].div(dep_df.sum(axis=1), axis=0) * 100
    dep_share.to_csv(RESULTS / "dependency_distribution.csv")

    demo_articles = [
        "The finance ministry said the economy expanded after exporters reported stronger demand and the central bank held interest rates steady.",
        "The club secured a dramatic victory after its striker scored twice in the final ten minutes of the championship match.",
    ]
    demo_rows = []
    for text in demo_articles:
        doc = nlp(text)
        demo_rows.append({
            "text": text,
            "predicted_category": best_pipe.predict([clean_text(text)])[0],
            "sentiment": sia.polarity_scores(text)["compound"],
            "entities": [{"text": e.text, "label": e.label_} for e in doc.ents],
            "subject_verb_object": [{"token": t.text, "dependency": t.dep_, "head": t.head.text} for t in doc if t.dep_ in {"nsubj", "dobj", "pobj", "ROOT"}],
        })
    (RESULTS / "demo_predictions.json").write_text(json.dumps(demo_rows, indent=2), encoding="utf-8")

    summary = {
        "source_rows": raw_rows, "analysis_rows": len(df), "missing_rows_removed": missing,
        "duplicate_articles_removed": duplicates, "categories": counts.to_dict(),
        "train_rows": len(X_train), "test_rows": len(X_test), "random_seed": SEED,
        "best_model": best_name, "best_accuracy": float(metrics.iloc[0]["accuracy"]),
        "best_macro_f1": float(metrics.iloc[0]["macro_f1"]),
        "vocabulary_size": int(len(best_pipe.named_steps["tfidf"].get_feature_names_out())),
        "mean_words_per_article": float(df["word_count"].mean()),
        "median_words_per_article": float(df["word_count"].median()),
        "sentiment_means": dict(zip(sent["category"], sent["mean"])),
        "entity_sample_articles": int(len(sample)),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df[["ArticleId", "content", "category", "clean_content", "word_count", "sentiment", "sentiment_label"]].to_csv(RESULTS / "article_analysis.csv", index=False)
    return summary


class NewsBot:
    """Integrated inference interface for category, sentiment, entities, POS and syntax."""

    def __init__(self, project_dir: str | Path = PROJECT):
        self.project_dir = Path(project_dir)
        self.classifier = joblib.load(self.project_dir / "models" / "newsbot_classifier.joblib")
        nltk_dir = self.project_dir / "work" / "nltk_data"
        nltk_dir.mkdir(parents=True, exist_ok=True)
        nltk.download("vader_lexicon", download_dir=str(nltk_dir), quiet=True)
        nltk.data.path.insert(0, str(nltk_dir))
        self.sentiment = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, article: str) -> dict:
        cleaned = clean_text(article)
        doc = self.nlp(cleaned)
        score = self.sentiment.polarity_scores(cleaned)["compound"]
        return {
            "category": self.classifier.predict([cleaned])[0],
            "sentiment_compound": score,
            "sentiment_label": "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral",
            "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "MONEY"}],
            "pos_counts": dict(Counter(tok.pos_ for tok in doc if not tok.is_space and not tok.is_punct)),
            "key_dependencies": [{"token": tok.text, "dependency": tok.dep_, "head": tok.head.text} for tok in doc if tok.dep_ in {"nsubj", "dobj", "pobj", "ROOT"}],
        }


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
