import os
from dotenv import load_dotenv

load_dotenv()
import sys
import csv
import ast
import json
from pprint import pp
from tqdm import tqdm
import pandas as pd
from urllib.parse import urlparse
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

pio.defaults.mathjax = None
from utils import *
from data_utils import *
from paper import with_paper_style, styler
from data_extraction import load_web_data_from_file
import nltk
import spacy
from nltk.stem.snowball import SnowballStemmer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from langdetect import detect
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from openai import OpenAI
from evaluator_prompts import *


CONF = "emnlp/query_reformulations"


SPACY_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
}

SNOWBALL_LANGS = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "nl": "dutch",
    "ru": "russian",
    "pt": "portuguese",
}

# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@lru_cache(maxsize=None)
def get_spacy_model(lang_code):
    if lang_code in SPACY_MODELS:
        return spacy.load(SPACY_MODELS[lang_code], disable=["parser", "ner"])
    return None


@lru_cache(maxsize=None)
def get_stemmer(lang_code):
    if lang_code in SNOWBALL_LANGS:
        return SnowballStemmer(SNOWBALL_LANGS[lang_code])
    return None


def preprocess_text(text, stem=False):
    try:
        lang = detect(text)
    except:
        lang = "en"

    nlp = get_spacy_model(lang)
    if nlp:
        doc = nlp(text)
        lemmas = [
            t.lemma_.lower()
            for t in doc
            if not t.is_punct and not t.is_space and not t.is_stop
        ]
    else:
        tokens = word_tokenize(text)
        lemmas = [t.lower() for t in tokens if t.isalnum()]

    if stem:
        stemmer = get_stemmer(lang)
        if stemmer:
            tokens = [stemmer.stem(t) for t in lemmas]
        else:
            tokens = lemmas
    else:
        tokens = lemmas

    return tokens


def preprocess_text_in_chunks(text, stem=False, max_chunk_chars=100_000):
    text = "" if text is None else str(text)
    if len(text) <= max_chunk_chars:
        return preprocess_text(text, stem=stem)

    tokens = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            tokens.extend(preprocess_text(chunk, stem=stem))

        start = end
        while start < len(text) and text[start].isspace():
            start += 1

    return tokens


def preprocess_texts(texts, stem=False, max_chunk_chars=100_000):
    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [texts]

    tokens = []
    for text in texts:
        tokens.extend(
            preprocess_text_in_chunks(
                text,
                stem=stem,
                max_chunk_chars=max_chunk_chars,
            )
        )
    return tokens


def compute_bleu(reference_tokens, candidate_tokens, bleu_type="bleu4"):
    """
    bleu_type:
        - 'bleu1' : unigram only (recommended for short queries)
        - 'bleu4' : standard BLEU-4
    """

    smoothie = SmoothingFunction().method1

    if bleu_type == "bleu1":
        weights = (1.0, 0, 0, 0)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)

    return sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=weights,
        smoothing_function=smoothie,
    )


def unigram_f1(reference_tokens, candidate_tokens):
    ref_counts = Counter(reference_tokens)
    cand_counts = Counter(candidate_tokens)

    overlap = sum(min(count, ref_counts[word]) for word, count in cand_counts.items())

    precision = overlap / max(len(candidate_tokens), 1)
    recall = overlap / max(len(reference_tokens), 1)

    if precision + recall == 0:
        return precision, recall, 0.0

    return precision, recall, 2 * precision * recall / (precision + recall)


def compute_keyword_overlap(user_query, system_queries, bleu_type="bleu1"):
    all_bleus = []
    for system_query in system_queries:
        all_bleus.append(compute_bleu([user_query], system_query, bleu_type))
    return np.mean(all_bleus).item()


def compute_embedding_similarity(user_query, system_queries):
    embeddings = model.encode([user_query] + system_queries)

    user_emb = embeddings[0]
    system_embs = embeddings[1:]

    scores = cosine_similarity(user_emb.reshape(1, -1), system_embs)[0]

    return float(scores.mean())


def compute_unigram_f1(user_query, system_queries):
    f1s = []
    ps = []
    rs = []
    for system_query in system_queries:
        p, r, f1 = unigram_f1(user_query, system_query)
        f1s.append(f1)
        rs.append(r)
        ps.append(p)
    return np.mean(ps).item(), np.mean(rs).item(), np.mean(f1s).item()


def gather_query_reform_effective_factors(df):
    df["thoughts_list"] = [[]] * len(df)
    df["web_queries"] = [[]] * len(df)
    df["sources"] = [[]] * len(df)
    df["memories"] = [[]] * len(df)

    for i, row in tqdm(df.iterrows()):
        all_web_queries = []
        all_thoughts = []
        all_retrieved_sources = []
        all_memories = []

        msgs = json.loads(row["turn_msgs"])
        web_queries = []
        retrieved_sources = []
        thoughts = []
        memories = []

        for msg in msgs:
            # thoughts
            thinking_type = msg["content"].get("content_type", None)
            thinking_thoughts = msg["content"].get("thoughts", [])
            if thinking_type == "thoughts":
                for tt in thinking_thoughts:
                    thoughts.append(tt.get("content", ""))

            # retrieved sources
            retrieved = msg.get("metadata", {}).get("search_result_groups", [])
            for r in retrieved:
                entries = r.get("entries", [])
                for entry in entries:
                    url = entry.get("url", "")
                    domain = urlparse(entry["url"]).netloc.replace("www.", "")
                    title = entry.get("title", "")
                    snippet = entry.get("snippet", "")
                    retrieved_sources.append(f"{title}\n{domain}\n{snippet}")

            retrieved = msg.get("metadata", {}).get("image_results", [])
            for r in retrieved:
                domain = urlparse(entry["url"]).netloc.replace("www.", "")
                title = entry.get("title", "")
                snippet = entry.get("snippet", "")
                retrieved_sources.append(f"{title}\n{domain}\n{snippet}")

            # memories
            memory = (
                msg.get("metadata", {})
                .get("user_context_message_data", {})
                .get("about_user_message", "")
            )
            if memory:
                memories.append(memory)

            # web queries
            search_queries = msg.get("metadata", {}).get("search_queries", [])
            for search_query in search_queries:
                web_queries.append(search_query["q"])
            web_queries += (
                msg.get("metadata", {})
                .get("search_model_queries", {})
                .get("queries", [])
            )
            dedeup_web_queries = list(set(web_queries))
            if dedeup_web_queries:
                all_web_queries.append(dedeup_web_queries)
                all_thoughts.append(thoughts)
                all_retrieved_sources.append(retrieved_sources)
                all_memories.append(memories)
                web_queries = []
                retrieved_sources = []
                thoughts = []
                memories = []

        df.loc[i, "thoughts_list"] = json.dumps(all_thoughts)
        df.loc[i, "web_queries"] = json.dumps(all_web_queries)
        df.loc[i, "sources"] = json.dumps(all_retrieved_sources)
        df.loc[i, "memories"] = json.dumps(all_memories)

    df.drop(columns=["turn_msgs"], inplace=True)
    df.to_csv(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.csv",
        index=False,
    )
    df.reset_index().to_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )


def compute_semantic_and_syntactic_similarity():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )

    metric_cols = [
        "keyword_matching_bleu",
        "keyword_matching_bleu_w_history",
        "keyword_matching_f1",
        "keyword_matching_f1_w_history",
        "keyword_matching_precision",
        "keyword_matching_precision_w_history",
        "keyword_matching_recall",
        "keyword_matching_recall_w_history",
        "embedding_similarity",
        "embedding_similarity_w_history",
    ]
    for col in metric_cols:
        df[col] = np.nan

    user_query_length = []
    user_query_history_length = []
    search_query_length = []

    for i, row in tqdm(df.iterrows()):
        user_msg_history = row["user_msg_history"]
        if not user_msg_history:
            continue
        user_query = user_msg_history[-1]
        user_msg_history = " ".join(row["user_msg_history"])
        system_queries_ = json.loads(row["web_queries"])
        system_queries = [q for qs in system_queries_ for q in qs]
        if not system_queries:
            continue

        user_query_length.append(len(user_query))
        user_query_history_length.append(len(user_msg_history))
        search_query_length += [len(q) for q in system_queries]

        emb_overlap = compute_embedding_similarity(user_query, system_queries)
        emb_overlap_w_history = compute_embedding_similarity(
            user_msg_history, system_queries
        )

        user_query = preprocess_text(user_query)
        user_msg_history = preprocess_text(user_msg_history)
        system_queries = [preprocess_text(q) for q in system_queries]

        kw_bleu_overlap = compute_keyword_overlap(user_query, system_queries)
        kw_bleu_overlap_w_history = compute_keyword_overlap(
            user_msg_history, system_queries
        )

        kw_p_overlap, kw_r_overlap, kw_f1_overlap = compute_unigram_f1(
            user_query, system_queries
        )
        kw_p_overlap_w_history, kw_r_overlap_w_history, kw_f1_overlap_w_history = (
            compute_unigram_f1(user_msg_history, system_queries)
        )

        df.loc[i, "keyword_matching_bleu"] = kw_bleu_overlap
        df.loc[i, "keyword_matching_bleu_w_history"] = kw_bleu_overlap_w_history
        df.loc[i, "keyword_matching_f1"] = kw_f1_overlap
        df.loc[i, "keyword_matching_f1_w_history"] = kw_f1_overlap_w_history
        df.loc[i, "keyword_matching_precision"] = kw_p_overlap
        df.loc[i, "keyword_matching_precision_w_history"] = kw_p_overlap_w_history
        df.loc[i, "keyword_matching_recall"] = kw_r_overlap
        df.loc[i, "keyword_matching_recall_w_history"] = kw_r_overlap_w_history
        df.loc[i, "embedding_similarity"] = emb_overlap
        df.loc[i, "embedding_similarity_w_history"] = emb_overlap_w_history

    print("User Query:")
    print(np.mean(user_query_length))
    print(np.std(user_query_length))

    print("User Query with history:")
    print(np.mean(user_query_history_length))
    print(np.std(user_query_history_length))

    print("Search Query:")
    print(np.mean(search_query_length))
    print(np.std(search_query_length))

    print(len(df))
    df = df.dropna(subset=metric_cols).copy()
    df.reset_index(drop=True, inplace=True)
    print(len(df))
    df.to_csv(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v2.csv",
        index=False,
    )
    df.to_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    )


def plot_semantic_and_syntactic_similarity():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    )

    cols = [
        "keyword_matching_precision",
        "keyword_matching_precision_w_history",
        "embedding_similarity",
        "embedding_similarity_w_history",
    ]

    col_names = {
        "keyword_matching_precision": "Keyword Matching<br>with User Query",
        "keyword_matching_precision_w_history": "Keyword Matching<br>with User Query + Chat History",
        "embedding_similarity": "Semantic Similarity<br> with User Query",
        "embedding_similarity_w_history": "Semantic Similarity<br> with User Query + Chat History",
    }
    plot_df = df.dropna(subset=cols).copy()

    fig = go.Figure()

    for col in cols:
        fig.add_trace(
            go.Violin(
                y=plot_df[col],
                name=col_names[col],
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            )
        )

    fig.update_layout(
        yaxis_title="Score",
        xaxis_title="Metric",
    )
    name = "query_similarity_violin_plot"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{name}.html")
    fig = with_paper_style(fig, config=styler(18, 14))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=18))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{name}.pdf", format="pdf")


def web_query_tokens_source_detection():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )
    df = _add_web_query_token_source_columns(df)

    df.reset_index(drop=True, inplace=True)
    _save_dataframe(df, "query_reformulation_with_thought_src_mem_v3")


DEFAULT_WEB_QUERY_TOKEN_SOURCE_FACTOR_COLS = {
    "all_new_words_from_user_queries": "Latest<br>User Prompt",
    "all_new_words_from_assistant_queries": "Conversation<br>History",
    "all_new_words_from_memories": "Other<br>Conversations",
    "all_new_words_from_sources": "Retrieved<br>Results",
    "all_new_words_from_pk": "Parametric<br>Knowledge<br>[Potentially]",
}

REPLAY_WEB_QUERY_TOKEN_SOURCE_FACTOR_COLS = {
    "all_new_words_from_user_queries": "User Query",
    "all_new_words_from_pk": "Unknown",
}


def _plot_web_query_tokens_source_detection_from_df(
    df,
    factor_cols=None,
    base_file_name="web_query_token_source_detection",
    skip_source_for_one_loop=True,
):
    factor_cols = dict(factor_cols or DEFAULT_WEB_QUERY_TOKEN_SOURCE_FACTOR_COLS)
    user_prompt_col = "all_new_words_from_user_queries"
    history_col = "all_new_words_from_assistant_queries"
    memories_col = "all_new_words_from_memories"
    unexplained_col = "all_new_words_from_pk"

    df = df.copy()
    for col in {"web_queries", "memories"} | set(factor_cols.keys()):
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        df[col] = df[col].apply(lambda x: _safe_json_value(x, []))

    def _flatten_nested_texts(values):
        if not isinstance(values, list):
            values = [values]
        flat_values = []
        for item in values:
            if isinstance(item, list):
                flat_values.extend(item)
            else:
                flat_values.append(item)
        return [item.strip() for item in flat_values if isinstance(item, str) and item.strip()]

    num_rows_with_memories = int(
        df["memories"].apply(lambda items: len(_flatten_nested_texts(items)) > 0).sum()
    )
    num_memory_keyword_hits = int(
        df["all_new_words_from_memories"].apply(lambda items: len(items) > 0).sum()
    )

    print(f"Rows with memories: {num_rows_with_memories}")
    print(f"Memory keywords in web queries: {num_memory_keyword_hits}")

    plot_factor_cols = {
        col: label for col, label in factor_cols.items() if col != memories_col
    }

    plot_df = df.copy()
    plot_df[history_col] = plot_df.apply(
        lambda row: list(set(row[history_col]) | set(row[user_prompt_col])),
        axis=1,
    )
    plot_df[unexplained_col] = plot_df.apply(
        lambda row: list(set(row[unexplained_col]) | set(row[memories_col])),
        axis=1,
    )

    plot_df["num_web_query_words"] = plot_df.apply(
        lambda row: len(set().union(*(set(row[col]) for col in plot_factor_cols))),
        axis=1,
    )
    plot_df["num_loops"] = plot_df["web_queries"].apply(len)
    plot_df = plot_df[plot_df["num_web_query_words"] > 0].copy()

    for col in plot_factor_cols:
        rate_col = f"{col}_rate"
        plot_df[rate_col] = plot_df.apply(
            lambda row: len(row[col]) / row["num_web_query_words"]
            if row["num_web_query_words"] > 0
            else np.nan,
            axis=1,
        )

    def _plot_subset(subset_df, file_name, title_suffix):
        if len(subset_df) == 0:
            return

        fig = go.Figure()
        for i, (col, label) in enumerate(plot_factor_cols.items()):
            if (
                skip_source_for_one_loop
                and "1_loop" in file_name
                and col == "all_new_words_from_sources"
            ):
                continue
            fig.add_trace(
                go.Box(
                    y=subset_df[f"{col}_rate"],
                    name=label,
                    marker_color=px.colors.qualitative.Plotly[i],
                    line_color=px.colors.qualitative.Plotly[i],
                    showlegend=False,
                    boxmean=True,
                )
            )

        fig.update_layout(
            xaxis_title="Keyword Source",
            yaxis_title="Share of Web Query Keywords",
            title=title_suffix,
            xaxis=dict(
                tickangle=0,
            )
        )
        fig.update_yaxes(tickformat=".0%")

        os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(22, 14))
        fig.update_xaxes(tickfont=dict(size=20))
        fig.update_yaxes(tickfont=dict(size=22))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    def _plot_bar_subset(subset_df, file_name, title_suffix):
        if len(subset_df) == 0:
            return

        subset_factor_cols = {
            col: label
            for col, label in plot_factor_cols.items()
            if not (
                skip_source_for_one_loop
                and "1_loop" in file_name
                and col == "all_new_words_from_sources"
            )
        }
        factor_counts = {
            col: subset_df[col].apply(lambda words: len(set(words))).sum()
            for col in subset_factor_cols
        }
        total_count = sum(factor_counts.values())
        if total_count == 0:
            return

        labels = list(subset_factor_cols.values())
        shares = [factor_counts[col] / total_count for col in subset_factor_cols]
        colors = px.colors.qualitative.Plotly[: len(subset_factor_cols)]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=shares,
                text=[f"{share:.1%}" for share in shares],
                textposition="outside",
                marker_color=colors,
                showlegend=False,
            )
        )
        fig.update_layout(
            xaxis_title="Keyword Source",
            yaxis_title="Share of Web Query Keywords",
            title=title_suffix,
        )
        fig.update_yaxes(tickformat=".0%", range=[0, max(shares) * 1.15])

        os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 14))
        fig.update_xaxes(tickfont=dict(size=14))
        fig.update_yaxes(tickfont=dict(size=18))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    one_loop_df = plot_df[plot_df["num_loops"] == 1].copy()
    multi_loop_df = plot_df[plot_df["num_loops"] > 1].copy()

    _plot_subset(
        one_loop_df,
        f"{base_file_name}_1_loop",
        "1 Web Query Loop",
    )
    _plot_subset(
        multi_loop_df,
        f"{base_file_name}_multi_loop",
        "2+ Web Query Loops",
    )
    _plot_bar_subset(
        one_loop_df,
        f"{base_file_name}_bar_1_loop",
        "1 Web Query Loop",
    )
    _plot_bar_subset(
        multi_loop_df,
        f"{base_file_name}_bar_multi_loop",
        "2+ Web Query Loops",
    )


def plot_web_query_tokens_source_detection():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v3.pkl"
    )
    _plot_web_query_tokens_source_detection_from_df(df)


def plot_web_query_tokens_source_detection_over_time(
    factor_cols=None,
    file_name="web_query_token_source_detection_over_time",
    tick_interval_months=2,
):
    factor_cols = factor_cols or DEFAULT_WEB_QUERY_TOKEN_SOURCE_FACTOR_COLS
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v3.pkl"
    ).copy()

    for col in ["web_queries"] + list(factor_cols.keys()):
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        df[col] = df[col].apply(lambda x: _safe_json_value(x, []))

    df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["num_loops"] = df["web_queries"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    df["num_web_query_words"] = df.apply(
        lambda row: len(set().union(*(set(row[col]) for col in factor_cols))),
        axis=1,
    )
    df = df[df["num_web_query_words"] > 0].copy()
    if len(df) == 0:
        print("No valid rows for web query token source trend plot.")
        return {"1_loop": [], "multi_loop": []}

    for col in factor_cols:
        rate_col = f"{col}_rate"
        df[rate_col] = df.apply(
            lambda row: len(set(row[col])) / row["num_web_query_words"],
            axis=1,
        )

    def _plot_subset(subset_df, subset_file_name, subset_title, excluded_factor_cols=None):
        if len(subset_df) == 0:
            print(f"No valid rows for {subset_title.lower()} trend plot.")
            return []

        excluded_factor_cols = excluded_factor_cols or set()
        plot_factor_cols = {
            col: label
            for col, label in factor_cols.items()
            if col not in excluded_factor_cols
        }
        if len(plot_factor_cols) == 0:
            print(f"No factor columns left to plot for {subset_title.lower()}.")
            return []

        subset_df = subset_df.copy()
        subset_df["month"] = subset_df["time"].dt.to_period("M").dt.to_timestamp()
        agg_kwargs = {
            f"{col}_rate": (f"{col}_rate", "mean")
            for col in plot_factor_cols
        }
        agg_kwargs["num_turns"] = ("month", "size")
        monthly = (
            subset_df.groupby("month")
            .agg(**agg_kwargs)
            .reset_index()
            .sort_values("month")
        )

        fig = go.Figure()
        palette = px.colors.qualitative.Plotly
        for i, (col, label) in enumerate(plot_factor_cols.items()):
            hover_label = label.replace("<br>", " ")
            fig.add_trace(
                go.Scatter(
                    x=monthly["month"],
                    y=monthly[f"{col}_rate"],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=palette[i % len(palette)]),
                    marker=dict(color=palette[i % len(palette)]),
                    customdata=monthly["num_turns"],
                    hovertemplate=(
                        "Month: %{x|%b %Y}<br>"
                        f"{hover_label}: "
                        "%{y:.1%}<br>"
                        "Turns: %{customdata}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Share of Web Query Keywords",
            title=subset_title,
            xaxis=dict(
                tickmode="linear",
                dtick=f"M{tick_interval_months}",
                tickformat="%b %Y",
                tickangle=-45,
            ),
            margin=dict(b=90),
        )
        fig.update_yaxes(tickformat=".0%")
        os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{subset_file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 16), legend_pos=(0.9, 1.35))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{subset_file_name}.pdf", format="pdf")

        records = monthly.copy()
        records["month"] = records["month"].dt.strftime("%Y-%m")
        return records.to_dict(orient="records")

    one_loop_df = df[df["num_loops"] == 1].copy()
    multi_loop_df = df[df["num_loops"] > 1].copy()

    return {
        "1_loop": _plot_subset(
            one_loop_df,
            f"{file_name}_1_loop",
            "1 Web Query Loop",
            excluded_factor_cols={"all_new_words_from_sources"},
        ),
        "multi_loop": _plot_subset(
            multi_loop_df,
            f"{file_name}_multi_loop",
            "2+ Web Query Loops",
        ),
    }


def check_retrieved_source_effect_for_one_loop():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v3.pkl"
    )

    for col in [
        "web_queries",
        "all_new_words_from_sources",
        "all_new_words_from_user_queries",
        "all_new_words_from_assistant_queries",
        "all_new_words_from_thoughts",
        "all_new_words_from_memories",
        "all_new_words_from_pk",
    ]:
        df[col] = df[col].apply(
            lambda x: x
            if isinstance(x, list)
            else (json.loads(x) if isinstance(x, str) and x else [])
        )

    df["num_loops"] = df["web_queries"].apply(len)
    one_loop_df = df[df["num_loops"] == 1].copy()
    if len(one_loop_df) == 0:
        return {
            "num_one_loop_turns": 0,
            "num_with_retrieved_source_effect": 0,
            "num_without_retrieved_source_effect": 0,
            "rate_without_retrieved_source_effect": None,
        }

    one_loop_df["retrieved_source_effect"] = one_loop_df[
        "all_new_words_from_sources"
    ].apply(lambda x: len(x) > 0)

    result = {
        "num_one_loop_turns": int(len(one_loop_df)),
        "num_with_retrieved_source_effect": int(one_loop_df["retrieved_source_effect"].sum()),
        "num_without_retrieved_source_effect": int((~one_loop_df["retrieved_source_effect"]).sum()),
        "rate_without_retrieved_source_effect": float((~one_loop_df["retrieved_source_effect"]).mean()),
    }

    violating_df = one_loop_df[one_loop_df["retrieved_source_effect"]].copy()
    inspect_cols = [
        "user_id",
        "conv_id",
        "turn_id",
        "topic",
        "time",
        "web_queries",
        "all_new_words",
        "all_new_words_from_sources",
        "all_new_words_from_user_queries",
        "all_new_words_from_assistant_queries",
        "all_new_words_from_thoughts",
        "all_new_words_from_memories",
        "all_new_words_from_pk",
    ]
    violating_df = violating_df[inspect_cols].reset_index(drop=True)
    violating_df.to_csv(
        f"{OUTPUT_PATH}/metadata/one_loop_retrieved_source_effect_counterexamples.csv",
        index=False,
    )

    print(result)
    if len(violating_df) > 0:
        print(violating_df.to_dict(orient="records"))
    return result


def _plot_number_of_loops_histogram_from_df(
    df,
    number_file_name="number_of_query_reformulations",
    parallel_file_name="parallel_queries_by_query_reformulations",
    samples_file_name="number_of_query_reformulations_samples",
    sample_language="en",
    drop_zero_loop_rows=True,
    trend_file_name="number_of_query_reformulations_and_parallel_queries_over_time",
    trend_tick_interval_months=2,
):
    bucket_order = ([] if drop_zero_loop_rows else ["0"]) + ["1", "2", "3+"]
    parallel_plot_x_order = ["1", "2", "3+"]

    def _bucket_count(value):
        return "3+" if value >= 3 else str(value)

    def _parallel_plot_x_bucket(value):
        return "3+" if value >= 3 else str(value)

    count_n_hops = {}
    loop_samples = {}
    parallel_query_counts = {
        n_hops_bucket: {parallel_bucket: 0 for parallel_bucket in bucket_order}
        for n_hops_bucket in parallel_plot_x_order
    }
    trend_rows = []

    for _, row in tqdm(df.iterrows()):
        all_system_queries = _safe_json_value(row.get("web_queries"), [])
        all_thoughts = _safe_json_value(row.get("thoughts_list"), [])
        all_retrieved_sources = _safe_json_value(row.get("sources"), [])

        n_hops = len(all_system_queries)
        if drop_zero_loop_rows and n_hops == 0:
            continue

        count_n_hops[n_hops] = count_n_hops.get(n_hops, 0) + 1
        bucket = _bucket_count(n_hops)
        max_parallel_queries = max(
            (
                len(query_group) if isinstance(query_group, list) else 1
                for query_group in all_system_queries
            ),
            default=0,
        )
        num_parallel_queries = sum(
            (
                len(query_group) if isinstance(query_group, list) else 1
                for query_group in all_system_queries
            )
        )
        trend_rows.append(
            {
                "time": row.get("time"),
                "num_loops": n_hops,
                "num_parallel_queries": num_parallel_queries,
            }
        )
        if n_hops >= 1 and max_parallel_queries >= 1:
            parallel_query_counts[_parallel_plot_x_bucket(n_hops)][
                _bucket_count(max_parallel_queries)
            ] += 1
        sample_language_match = (
            sample_language is None or row.get("language") == sample_language
        )
        if n_hops >= 1 and bucket not in loop_samples and sample_language_match:
            user_msg_history = _safe_json_value(row.get("user_msg_history"), [])
            if isinstance(user_msg_history, str):
                user_msg_history = [user_msg_history]
            loop_samples[bucket] = {
                "num_loops_bucket": bucket,
                "num_loops": n_hops,
                "user_query": str(user_msg_history[-1]).strip() if user_msg_history else "",
                "web_queries": all_system_queries,
                "thoughts": all_thoughts,
                "srcs_retrieved": all_retrieved_sources,
                "result_key": row.get("result_key"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                # "topic": row.get("topic"),
            }

    print(count_n_hops)

    count_n_hops_sum = sum(count_n_hops.values())
    print(count_n_hops_sum)
    if count_n_hops_sum == 0:
        print("No web query loops to plot.")
        return {
            "count_n_hops": count_n_hops,
            "parallel_query_counts": parallel_query_counts,
            "loop_samples": loop_samples,
            "time_trend": [],
        }

    binned_percentages = {bucket: 0.0 for bucket in bucket_order}
    for n_hops, count in count_n_hops.items():
        percentage = 100 * count / count_n_hops_sum
        if n_hops >= 3:
            binned_percentages["3+"] += percentage
        elif n_hops >= 1:
            binned_percentages[str(n_hops)] += percentage
        elif "0" in binned_percentages:
            binned_percentages["0"] += percentage

    x = list(binned_percentages.keys())
    y = [round(binned_percentages[label], 2) for label in x]
    text = [f"{value:.1f}%" for value in y]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            text=text,
            textposition="outside",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Number of Query Formulations",
        yaxis_title="Turns (%)",
        yaxis=dict(range=[0, max(y) * 1.15 if y else 1]),
    )
    fig.update_yaxes(ticksuffix="%")
    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    file_name = number_file_name
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18))
    fig.write_image(
        f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf"
    )
    to_json(
        [
            loop_samples[label]
            for label in x
            if label in loop_samples
        ],
        f"{OUTPUT_PATH}/{CONF}/{samples_file_name}.json",
    )

    fig = go.Figure()
    for parallel_bucket in bucket_order:
        y = [
            round(
                100
                * parallel_query_counts[n_hops_bucket][parallel_bucket]
                / count_n_hops_sum,
                2,
            )
            for n_hops_bucket in parallel_plot_x_order
        ]
        fig.add_trace(
            go.Bar(
                x=parallel_plot_x_order,
                y=y,
                name=parallel_bucket,
                text=[f"{value:.1f}%" if value > 0 else "" for value in y],
                textposition="outside",
                hovertemplate=(
                    "Query formulations: %{x}<br>"
                    f"Max Fan-out queries: {parallel_bucket}<br>"
                    "Turns: %{customdata}<br>"
                    "Share: %{y:.2f}%<extra></extra>"
                ),
                customdata=[
                    parallel_query_counts[n_hops_bucket][parallel_bucket]
                    for n_hops_bucket in parallel_plot_x_order
                ],
            )
        )
    max_breakdown_y = max(
        [
            100 * count / count_n_hops_sum
            for n_hops_bucket in parallel_plot_x_order
            for count in parallel_query_counts[n_hops_bucket].values()
        ]
        or [0]
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Number of Query Formulations",
        yaxis_title="Turns (%)",
        yaxis=dict(range=[0, max_breakdown_y * 1.25 if max_breakdown_y else 1]),
        legend_title_text="Max Fan-out Queries",
    )
    fig.update_yaxes(ticksuffix="%")
    file_name = parallel_file_name
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18))
    fig.write_image(
        f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf"
    )

    trend_records = []
    if trend_file_name:
        trend_df = pd.DataFrame(trend_rows)
        if "time" in trend_df.columns:
            trend_df["time"] = pd.to_datetime(trend_df["time"], errors="coerce")
            trend_df = trend_df.dropna(subset=["time"])

        if len(trend_df) > 0:
            trend_df["month"] = trend_df["time"].dt.to_period("M").dt.to_timestamp()
            monthly_trends = (
                trend_df.groupby("month")
                .agg(
                    avg_num_loops=("num_loops", "mean"),
                    avg_num_parallel_queries=("num_parallel_queries", "mean"),
                    num_turns=("num_loops", "size"),
                )
                .reset_index()
                .sort_values("month")
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=monthly_trends["month"],
                    y=monthly_trends["avg_num_loops"],
                    mode="lines+markers",
                    name="Number of Iterations",
                    customdata=monthly_trends["num_turns"],
                    hovertemplate=(
                        "Month: %{x|%b %Y}<br>"
                        "Average: %{y:.2f}<br>"
                        "Turns: %{customdata}<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=monthly_trends["month"],
                    y=monthly_trends["avg_num_parallel_queries"],
                    mode="lines+markers",
                    name="Number of Fan-out Queries",
                    customdata=monthly_trends["num_turns"],
                    hovertemplate=(
                        "Month: %{x|%b %Y}<br>"
                        "Average: %{y:.2f}<br>"
                        "Turns: %{customdata}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Average Number",
                xaxis=dict(
                    tickmode="linear",
                    dtick=f"M{trend_tick_interval_months}",
                    tickformat="%b %Y",
                    tickangle=-45,
                ),
                margin=dict(b=90),
            )
            file_name = trend_file_name
            fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
            fig = with_paper_style(fig, config=styler(18, 16), legend_pos=(0.9, 1.2))
            fig.write_image(
                f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf"
            )

            trend_records = monthly_trends.copy()
            trend_records["month"] = trend_records["month"].dt.strftime("%Y-%m")
            trend_records = trend_records.to_dict(orient="records")
        else:
            print("No valid timestamps found for loop trend plot.")

    return {
        "count_n_hops": count_n_hops,
        "parallel_query_counts": parallel_query_counts,
        "loop_samples": loop_samples,
        "time_trend": trend_records,
    }


def plot_number_of_loops_histogram():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    )
    return _plot_number_of_loops_histogram_from_df(df)


def plot_number_of_query_reformulations_over_time(
    drop_zero_loop_rows=True,
    file_name="number_of_query_reformulations_over_time",
    tick_interval_months=2,
):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    ).copy()

    if "web_queries" not in df.columns:
        print("Column `web_queries` not found.")
        return []

    df["web_queries"] = df["web_queries"].apply(lambda x: _safe_json_value(x, []))
    df["num_loops"] = df["web_queries"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if drop_zero_loop_rows:
        df = df[df["num_loops"] > 0].copy()

    df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    if len(df) == 0:
        print("No valid rows for query reformulation trend plot.")
        return []

    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month")
        .agg(
            avg_num_loops=("num_loops", "mean"),
            num_turns=("num_loops", "size"),
        )
        .reset_index()
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_num_loops"],
            mode="lines+markers",
            name="Number of Query Formulations",
            customdata=monthly["num_turns"],
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Average: %{y:.2f}<br>"
                "Turns: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Number of Query Formulations",
        xaxis=dict(
            tickmode="linear",
            dtick=f"M{tick_interval_months}",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        margin=dict(b=90),
    )
    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    records = monthly.copy()
    records["month"] = records["month"].dt.strftime("%Y-%m")
    return records.to_dict(orient="records")


def plot_number_of_parallel_queries_over_time(
    drop_zero_loop_rows=True,
    file_name="number_of_parallel_queries_over_time",
    tick_interval_months=2,
):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    ).copy()

    if "web_queries" not in df.columns:
        print("Column `web_queries` not found.")
        return []

    df["web_queries"] = df["web_queries"].apply(lambda x: _safe_json_value(x, []))
    df["num_loops"] = df["web_queries"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if drop_zero_loop_rows:
        df = df[df["num_loops"] > 0].copy()

    df["num_parallel_queries"] = df["web_queries"].apply(
        lambda queries: sum(
            len(query_group) if isinstance(query_group, list) else 1
            for query_group in queries
        )
        if isinstance(queries, list)
        else 0
    )

    df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    if len(df) == 0:
        print("No valid rows for parallel query trend plot.")
        return []

    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month")
        .agg(
            avg_num_parallel_queries=("num_parallel_queries", "mean"),
            num_turns=("num_parallel_queries", "size"),
        )
        .reset_index()
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_num_parallel_queries"],
            mode="lines+markers",
            name="Number of Parallel Queries",
            customdata=monthly["num_turns"],
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Average: %{y:.2f}<br>"
                "Turns: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Number of Parallel Queries",
        xaxis=dict(
            tickmode="linear",
            dtick=f"M{tick_interval_months}",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        margin=dict(b=90),
    )
    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    records = monthly.copy()
    records["month"] = records["month"].dt.strftime("%Y-%m")
    return records.to_dict(orient="records")


@lru_cache(maxsize=200_000)
def _count_terms(text, remove_stopwords=False):
    text = "" if text is None else str(text).strip()
    if not text:
        return 0
    if remove_stopwords:
        return len(preprocess_text_in_chunks(text))
    return len(text.split())


def _clean_web_query_groups(value):
    parsed = _safe_json_value(value, [])
    if not isinstance(parsed, list):
        return []

    cleaned_groups = []
    for query_group in parsed:
        if isinstance(query_group, list):
            cleaned_group = [
                str(query).strip()
                for query in query_group
                if isinstance(query, str) and str(query).strip()
            ]
        elif isinstance(query_group, str) and query_group.strip():
            cleaned_group = [query_group.strip()]
        else:
            cleaned_group = []

        if cleaned_group:
            cleaned_groups.append(cleaned_group)

    return cleaned_groups


def _iteration_bucket(value):
    if value <= 1:
        return "1 iteration"
    if value == 2:
        return "2 iterations"
    return "3+ iterations"


def _fanout_bucket(value):
    if value <= 1:
        return "1 fanout"
    if value == 2:
        return "2 fanout"
    return "3+ fanout"


def _hex_to_rgba(hex_color, alpha=0.16):
    if isinstance(hex_color, str) and hex_color.startswith("#") and len(hex_color) == 7:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(99, 110, 250, {alpha})"


def _plot_term_count_over_time_by_bucket(
    df,
    *,
    value_col,
    bucket_col,
    bucket_order,
    legend_title,
    title,
    file_name,
    tick_interval_months=2,
    yaxis_max=None,
):
    plot_df = df.dropna(subset=["month", value_col, bucket_col]).copy()
    if len(plot_df) == 0:
        print(f"No valid rows for `{file_name}`.")
        return []

    monthly = (
        plot_df.groupby(["month", bucket_col])[value_col]
        .agg(mean="mean", std="std", num_turns="size")
        .reset_index()
        .sort_values("month")
    )
    monthly["std"] = monthly["std"].fillna(0.0)
    monthly["se"] = monthly["std"] / np.sqrt(monthly["num_turns"].clip(lower=1))
    monthly["lower"] = (monthly["mean"] - monthly["se"]).clip(lower=0)
    monthly["upper"] = monthly["mean"] + monthly["se"]

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for idx, bucket in enumerate(bucket_order):
        bucket_df = monthly[monthly[bucket_col] == bucket].sort_values("month")
        if len(bucket_df) == 0:
            continue

        color = palette[idx % len(palette)]
        fill_color = _hex_to_rgba(color, alpha=0.16)

        fig.add_trace(
            go.Scatter(
                x=bucket_df["month"],
                y=bucket_df["lower"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bucket_df["month"],
                y=bucket_df["upper"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        customdata = np.column_stack(
            (
                bucket_df["num_turns"].to_numpy(dtype=float),
                bucket_df["lower"].to_numpy(dtype=float),
                bucket_df["upper"].to_numpy(dtype=float),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bucket_df["month"],
                y=bucket_df["mean"],
                mode="lines+markers",
                name=bucket,
                line=dict(color=color, width=2.5),
                marker=dict(color=color, size=6),
                customdata=customdata,
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Bucket: %{fullData.name}<br>"
                    "Average terms: %{y:.2f}<br>"
                    "Error band (mean +/- SE): "
                    "[%{customdata[1]:.2f}, %{customdata[2]:.2f}]<br>"
                    "Turns: %{customdata[0]:.0f}<extra></extra>"
                ),
            )
        )

    if len(fig.data) == 0:
        print(f"No bucket series to plot for `{file_name}`.")
        return []

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Number of Query Terms",
        title=title,
        legend_title_text=legend_title,
        xaxis=dict(
            tickmode="linear",
            dtick=f"M{tick_interval_months}",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    if yaxis_max is not None:
        fig.update_yaxes(range=[0, yaxis_max])
    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16), legend_pos=(0.9, 1.2))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    records = monthly.copy()
    records["month"] = records["month"].dt.strftime("%Y-%m")
    return records.to_dict(orient="records")


def plot_query_term_count_trends_over_time(remove_stopwords=False):
    return _plot_query_term_count_trends_over_time_multiplatform(
        remove_stopwords=remove_stopwords
    )


def _plot_query_term_count_trends_over_time_multiplatform(remove_stopwords=False):
    platform_configs = [
        ("openai", "OpenAI"),
        ("claude", "Claude"),
        ("grok", "Grok"),
        # ("deepseek", "DeepSeek"),
    ]
    iteration_bucket_order = ["1", "2", "3+"]

    def _timeline_iteration_bucket(num_iterations):
        if num_iterations <= 1:
            return "1"
        if num_iterations == 2:
            return "2"
        return "3+"

    def _platform_candidate_paths(platform):
        if platform == "openai":
            return [
                f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v2.pkl",
            ]

        candidates = [
            f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem.pkl",
        ]
        return candidates

    def _load_platform_df(platform):
        for candidate_path in _platform_candidate_paths(platform):
            if not os.path.exists(candidate_path):
                continue
            try:
                return pd.read_pickle(candidate_path).copy(), candidate_path
            except Exception as e:
                print(f"Failed to load `{candidate_path}`: {e}")

        print(f"No query reformulation metadata file found for `{platform}`.")
        return None, None

    def _collect_platform_metrics(df):
        df = df.copy()
        df["time"] = pd.to_datetime(df.get("time"), errors="coerce")

        metrics = {
            "terms_per_query": [],
            "terms_per_prompt": [],
            "avg_query_terms_per_prompt": [],
            "total_query_terms_per_prompt": [],
            "total_queries_per_prompt": [],
            "iterations_per_prompt": [],
            "iteration_timeline_rows": [],
        }

        for _, row in tqdm(df.iterrows(), total=len(df)):
            web_query_groups = _clean_web_query_groups(row.get("web_queries"))
            if not web_query_groups:
                continue

            flat_web_queries = [q for group in web_query_groups for q in group]
            if len(flat_web_queries) == 0:
                continue

            web_query_term_counts = [
                _count_terms(query, remove_stopwords) for query in flat_web_queries
            ]
            if len(web_query_term_counts) == 0:
                continue

            metrics["terms_per_query"] += web_query_term_counts
            latest_user_query = _row_latest_user_query(row)
            metrics["terms_per_prompt"].append(
                _count_terms(latest_user_query, remove_stopwords)
            )
            metrics["avg_query_terms_per_prompt"].append(
                float(np.mean(web_query_term_counts))
            )
            metrics["total_query_terms_per_prompt"].append(
                float(np.sum(web_query_term_counts))
            )
            metrics["total_queries_per_prompt"].append(len(flat_web_queries))
            num_iterations = int(len(web_query_groups))
            metrics["iterations_per_prompt"].append(num_iterations)

            row_time = row.get("time")
            if pd.notna(row_time):
                metrics["iteration_timeline_rows"].append(
                    {
                        "month": row_time.to_period("M").to_timestamp(),
                        "iteration_bucket": _timeline_iteration_bucket(num_iterations),
                    }
                )

        return metrics

    def _sorted_cdf(values):
        sorted_values = np.sort(np.asarray(values, dtype=float))
        cdf_values = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
        return sorted_values, cdf_values

    platform_metrics = {}
    platform_data_sources = {}
    for platform, _ in platform_configs:
        platform_df, source_path = _load_platform_df(platform)
        if platform_df is None:
            continue
        if "web_queries" not in platform_df.columns:
            print(f"Column `web_queries` not found for `{platform}`.")
            continue

        metrics = _collect_platform_metrics(platform_df)
        if len(metrics["terms_per_query"]) == 0 or len(metrics["terms_per_prompt"]) == 0:
            print(f"No valid rows with web queries/user prompts for `{platform}`.")
            continue

        platform_metrics[platform] = metrics
        platform_data_sources[platform] = source_path

    if len(platform_metrics) == 0:
        print("No valid rows found for any platform in query-term CDF plots.")
        return {}

    query_complexity_output_dir = f"{OUTPUT_PATH}/{CONF}/query_complexity"
    os.makedirs(query_complexity_output_dir, exist_ok=True)
    palette = px.colors.qualitative.Plotly

    def _plot_cdf_by_platform(
        metric_key,
        *,
        value_col,
        xaxis_title,
        file_name,
        hover_label,
        x_fmt=".2f",
        xaxis_config=None,
    ):
        def _format_tick(value):
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            return f"{value:g}"

        def _nice_tick_step(span):
            if span <= 0:
                return 1.0
            raw = span / 5.0
            power = 10 ** np.floor(np.log10(raw))
            ratio = raw / power
            if ratio <= 1:
                nice = 1
            elif ratio <= 2:
                nice = 2
            elif ratio <= 5:
                nice = 5
            else:
                nice = 10
            return float(nice * power)

        fig = go.Figure()
        points_by_platform = {}

        for idx, (platform, display_name) in enumerate(platform_configs):
            if platform not in platform_metrics:
                continue

            values = platform_metrics[platform][metric_key]
            if len(values) == 0:
                continue

            sorted_values, cdf_values = _sorted_cdf(values)
            points_by_platform[platform] = pd.DataFrame(
                {value_col: sorted_values, "cdf": cdf_values}
            ).to_dict(orient="records")

            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=sorted_values,
                    y=cdf_values,
                    mode="lines",
                    name=display_name,
                    line=dict(width=2.5, color=color),
                    hovertemplate=(
                        f"{hover_label}: %{{x:{x_fmt}}}<br>"
                        "CDF: %{y:.3f}"
                        f"<extra>{display_name}</extra>"
                    ),
                )
            )

        layout_kwargs = {
            "xaxis_title": xaxis_title,
            "yaxis_title": "Cumulative Probability",
            "yaxis": dict(range=[0, 1]),
            "margin": dict(t=5),
        }
        if xaxis_config is not None:
            xaxis_settings = dict(xaxis_config)
            range_values = xaxis_settings.get("range")
            if (
                isinstance(range_values, (list, tuple))
                and len(range_values) == 2
            ):
                try:
                    range_start = float(range_values[0])
                    range_end = float(range_values[1])
                    if np.isfinite(range_start) and np.isfinite(range_end) and range_end > range_start:
                        tick_step = _nice_tick_step(range_end - range_start)
                        tick_values = np.arange(
                            range_start, range_end + (tick_step * 0.5), tick_step
                        )
                        if len(tick_values) == 0:
                            tick_values = np.array([range_start, range_end], dtype=float)
                        if tick_values[-1] < (range_end - 1e-9):
                            tick_values = np.append(tick_values, range_end)
                        else:
                            tick_values[-1] = range_end
                        tick_values = np.unique(np.round(tick_values, 10))
                        tick_text = [_format_tick(v) for v in tick_values]
                        if len(tick_text) > 0:
                            tick_text[-1] = f"{tick_text[-1]}+"
                        xaxis_settings.update(
                            {
                                "tickmode": "array",
                                "tickvals": tick_values.tolist(),
                                "ticktext": tick_text,
                            }
                        )
                except (TypeError, ValueError):
                    pass
            layout_kwargs["xaxis"] = xaxis_settings
        fig.update_layout(**layout_kwargs)

        fig.write_html(f"{query_complexity_output_dir}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(26, 26))
        fig.write_image(f"{query_complexity_output_dir}/{file_name}.pdf", format="pdf")
        return points_by_platform

    web_file_name = "web_query_terms_cdf"
    user_file_name = "user_prompt_terms_cdf"
    avg_query_terms_per_prompt_file_name = "avg_web_query_terms_per_prompt_cdf"
    total_query_terms_per_prompt_file_name = "total_web_query_terms_per_prompt_cdf"
    total_queries_file_name = "total_web_queries_per_prompt_cdf"
    iterations_file_name = "iterations_per_prompt_cdf"
    timeline_file_name = "web_prompts_by_iteration_over_time"

    web_query_cdf_points_by_platform = _plot_cdf_by_platform(
        "terms_per_query",
        value_col="web_query_terms",
        xaxis_title="Number of Terms",
        file_name=web_file_name,
        hover_label="Web query terms",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    user_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "terms_per_prompt",
        value_col="user_prompt_terms",
        xaxis_title="Number of Terms",
        file_name=user_file_name,
        hover_label="User prompt terms",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    avg_web_query_terms_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "avg_query_terms_per_prompt",
        value_col="avg_web_query_terms_per_prompt",
        xaxis_title="Average Number of Web Query Terms Per User Prompt",
        file_name=avg_query_terms_per_prompt_file_name,
        hover_label="Avg web query terms per prompt",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    total_web_query_terms_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "total_query_terms_per_prompt",
        value_col="total_web_query_terms_per_prompt",
        xaxis_title="Total Number of Web Query Terms Per User Prompt",
        file_name=total_query_terms_per_prompt_file_name,
        hover_label="Total web query terms per prompt",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    total_web_queries_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "total_queries_per_prompt",
        value_col="total_web_queries_per_prompt",
        xaxis_title="Number of Web Queries Per User Prompt",
        file_name=total_queries_file_name,
        hover_label="Total web queries per prompt",
        x_fmt=".0f",
        xaxis_config=dict(range=[0, 10]),
    )
    iterations_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "iterations_per_prompt",
        value_col="iterations_per_prompt",
        xaxis_title="Number of Iterations Per User Prompt",
        file_name=iterations_file_name,
        hover_label="Iterations per prompt",
        x_fmt=".0f",
        xaxis_config=dict(range=[0, 10]),
    )

    timeline_plot_files_by_platform = {}
    timeline_points_by_platform = {}
    for platform, display_name in platform_configs:
        if platform not in platform_metrics:
            continue

        iteration_timeline_df = pd.DataFrame(
            platform_metrics[platform]["iteration_timeline_rows"]
        )
        if len(iteration_timeline_df) == 0:
            continue

        monthly_iteration_counts = (
            iteration_timeline_df.groupby(["month", "iteration_bucket"])
            .size()
            .reset_index(name="num_prompts")
        )
        monthly_iteration_counts["iteration_bucket"] = pd.Categorical(
            monthly_iteration_counts["iteration_bucket"],
            categories=iteration_bucket_order,
            ordered=True,
        )
        months = sorted(monthly_iteration_counts["month"].dropna().unique().tolist())
        full_index = pd.MultiIndex.from_product(
            [months, iteration_bucket_order],
            names=["month", "iteration_bucket"],
        )
        monthly_iteration_counts = (
            monthly_iteration_counts.set_index(["month", "iteration_bucket"])
            .reindex(full_index, fill_value=0)
            .reset_index()
            .sort_values(["month", "iteration_bucket"])
        )
        monthly_iteration_counts["month_total_prompts"] = monthly_iteration_counts.groupby(
            "month"
        )["num_prompts"].transform("sum")
        monthly_iteration_counts["pct_prompts"] = np.where(
            monthly_iteration_counts["month_total_prompts"] > 0,
            (monthly_iteration_counts["num_prompts"] * 100.0)
            / monthly_iteration_counts["month_total_prompts"],
            0.0,
        )

        timeline_fig = go.Figure()
        iteration_bucket_display = {
            "1": "1 iteration",
            "2": "2 iterations",
            "3+": "3+ iterations",
        }
        for iteration_bucket in iteration_bucket_order:
            subset = monthly_iteration_counts[
                monthly_iteration_counts["iteration_bucket"] == iteration_bucket
            ].sort_values("month")
            customdata = np.column_stack(
                (
                    subset["num_prompts"].to_numpy(dtype=float),
                    subset["month_total_prompts"].to_numpy(dtype=float),
                )
            )
            timeline_fig.add_trace(
                go.Scatter(
                    x=subset["month"],
                    y=subset["pct_prompts"],
                    mode="lines+markers",
                    name=iteration_bucket_display.get(iteration_bucket, iteration_bucket),
                    customdata=customdata,
                    hovertemplate=(
                        "Month: %{x|%b %Y}<br>"
                        "Iteration group: %{fullData.name}<br>"
                        "Share of web-search prompts: %{y:.1f}%<br>"
                        "Prompts in group: %{customdata[0]:.0f}<br>"
                        "Total web-search prompts: %{customdata[1]:.0f}<extra></extra>"
                    ),
                )
            )

        timeline_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Share of User Prompts",
            title=f"User Prompts by Iteration Bucket Over Time ({display_name})",
            xaxis=dict(
                tickmode="linear",
                dtick="M1",
                tickformat="%b %Y",
                tickangle=-45,
            ),
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            margin=dict(b=90),
        )
        platform_timeline_file_name = f"{timeline_file_name}_{platform}"
        timeline_fig.write_html(
            f"{query_complexity_output_dir}/{platform_timeline_file_name}.html"
        )
        timeline_fig = with_paper_style(timeline_fig, config=styler(18, 16))
        timeline_fig.write_image(
            f"{query_complexity_output_dir}/{platform_timeline_file_name}.pdf", format="pdf"
        )

        timeline_records = monthly_iteration_counts.copy()
        timeline_records["month"] = timeline_records["month"].dt.strftime("%Y-%m")
        timeline_records["iteration_bucket"] = timeline_records["iteration_bucket"].astype(
            str
        )
        timeline_points_by_platform[platform] = timeline_records.to_dict(orient="records")
        timeline_plot_files_by_platform[platform] = platform_timeline_file_name

    summary_by_platform = {}
    for platform, metrics in platform_metrics.items():
        summary_by_platform[platform] = {
            "num_web_queries": int(len(metrics["terms_per_query"])),
            "num_prompts": int(len(metrics["terms_per_prompt"])),
            "mean_web_query_terms": float(np.mean(metrics["terms_per_query"])),
            "median_web_query_terms": float(np.median(metrics["terms_per_query"])),
            "mean_user_prompt_terms": float(np.mean(metrics["terms_per_prompt"])),
            "median_user_prompt_terms": float(np.median(metrics["terms_per_prompt"])),
            "mean_avg_web_query_terms_per_prompt": float(
                np.mean(metrics["avg_query_terms_per_prompt"])
            ),
            "median_avg_web_query_terms_per_prompt": float(
                np.median(metrics["avg_query_terms_per_prompt"])
            ),
            "mean_total_web_query_terms_per_prompt": float(
                np.mean(metrics["total_query_terms_per_prompt"])
            ),
            "median_total_web_query_terms_per_prompt": float(
                np.median(metrics["total_query_terms_per_prompt"])
            ),
            "mean_total_web_queries_per_prompt": float(
                np.mean(metrics["total_queries_per_prompt"])
            ),
            "median_total_web_queries_per_prompt": float(
                np.median(metrics["total_queries_per_prompt"])
            ),
            "mean_iterations_per_prompt": float(np.mean(metrics["iterations_per_prompt"])),
            "median_iterations_per_prompt": float(
                np.median(metrics["iterations_per_prompt"])
            ),
        }

    primary_platform = "openai"
    if primary_platform not in platform_metrics:
        primary_platform = next(iter(platform_metrics.keys()))
    primary_summary = summary_by_platform[primary_platform]
    primary_timeline_points = timeline_points_by_platform.get(primary_platform, [])
    primary_timeline_file = timeline_plot_files_by_platform.get(primary_platform)

    return {
        "platforms_plotted": list(platform_metrics.keys()),
        "platform_data_sources": platform_data_sources,
        "web_query_cdf_points_by_platform": web_query_cdf_points_by_platform,
        "user_prompt_cdf_points_by_platform": user_prompt_cdf_points_by_platform,
        "avg_web_query_terms_per_prompt_cdf_points_by_platform": avg_web_query_terms_per_prompt_cdf_points_by_platform,
        "total_web_query_terms_per_prompt_cdf_points_by_platform": total_web_query_terms_per_prompt_cdf_points_by_platform,
        "total_web_queries_per_prompt_cdf_points_by_platform": total_web_queries_per_prompt_cdf_points_by_platform,
        "iterations_per_prompt_cdf_points_by_platform": iterations_per_prompt_cdf_points_by_platform,
        "web_query_cdf_points": web_query_cdf_points_by_platform.get(primary_platform, []),
        "user_prompt_cdf_points": user_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "avg_web_query_terms_per_prompt_cdf_points": avg_web_query_terms_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "total_web_query_terms_per_prompt_cdf_points": total_web_query_terms_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "total_web_queries_per_prompt_cdf_points": total_web_queries_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "iterations_per_prompt_cdf_points": iterations_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "web_prompts_by_iteration_over_time_points": primary_timeline_points,
        "web_prompts_by_iteration_over_time_points_by_platform": timeline_points_by_platform,
        "summary_by_platform": summary_by_platform,
        "num_web_queries": primary_summary["num_web_queries"],
        "num_prompts": primary_summary["num_prompts"],
        "web_query_plot_file": web_file_name,
        "user_prompt_plot_file": user_file_name,
        "avg_web_query_terms_per_prompt_plot_file": avg_query_terms_per_prompt_file_name,
        "total_web_query_terms_per_prompt_plot_file": total_query_terms_per_prompt_file_name,
        "total_web_queries_per_prompt_plot_file": total_queries_file_name,
        "iterations_per_prompt_plot_file": iterations_file_name,
        "web_prompts_by_iteration_over_time_plot_file": primary_timeline_file,
        "web_prompts_by_iteration_over_time_plot_files_by_platform": timeline_plot_files_by_platform,
        "timeline_platform": primary_platform,
        "mean_web_query_terms": primary_summary["mean_web_query_terms"],
        "median_web_query_terms": primary_summary["median_web_query_terms"],
        "mean_user_prompt_terms": primary_summary["mean_user_prompt_terms"],
        "median_user_prompt_terms": primary_summary["median_user_prompt_terms"],
        "mean_avg_web_query_terms_per_prompt": primary_summary[
            "mean_avg_web_query_terms_per_prompt"
        ],
        "median_avg_web_query_terms_per_prompt": primary_summary[
            "median_avg_web_query_terms_per_prompt"
        ],
        "mean_total_web_query_terms_per_prompt": primary_summary[
            "mean_total_web_query_terms_per_prompt"
        ],
        "median_total_web_query_terms_per_prompt": primary_summary[
            "median_total_web_query_terms_per_prompt"
        ],
        "mean_total_web_queries_per_prompt": primary_summary[
            "mean_total_web_queries_per_prompt"
        ],
        "median_total_web_queries_per_prompt": primary_summary[
            "median_total_web_queries_per_prompt"
        ],
        "mean_iterations_per_prompt": primary_summary["mean_iterations_per_prompt"],
        "median_iterations_per_prompt": primary_summary["median_iterations_per_prompt"],
    }


def distribution_of_web_query_and_thoughts_over_time():
    df_all_turns = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )
    df_with_web_query = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    )

    def has_non_empty_nested_list(value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return False
        if not isinstance(value, list):
            return False
        return any(isinstance(item, list) and len(item) > 0 for item in value)

    df_all_turns = df_all_turns.copy()
    df_with_web_query = df_with_web_query.copy()

    df_all_turns["time"] = pd.to_datetime(df_all_turns["time"], errors="coerce")
    df_with_web_query["time"] = pd.to_datetime(df_with_web_query["time"], errors="coerce")
    df_all_turns["month"] = df_all_turns["time"].dt.to_period("M").dt.to_timestamp()
    df_with_web_query["month"] = df_with_web_query["time"].dt.to_period("M").dt.to_timestamp()

    monthly_total_turns = (
        df_all_turns.groupby("month")["turn_id"]
        .nunique()
        .reset_index(name="total_turns")
    )
    monthly_web_query_turns = (
        df_with_web_query.groupby("month")["turn_id"]
        .nunique()
        .reset_index(name="web_query_turns")
    )
    monthly_thought_turns = (
        df_with_web_query[
            df_with_web_query["thoughts_list"].apply(has_non_empty_nested_list)
        ]
        .groupby("month")["turn_id"]
        .nunique()
        .reset_index(name="thought_turns")
    )
    monthly = (
        monthly_total_turns
        .merge(monthly_web_query_turns, on="month", how="left")
        .merge(monthly_thought_turns, on="month", how="left")
        .sort_values("month")
    )
    monthly[["web_query_turns", "thought_turns"]] = monthly[
        ["web_query_turns", "thought_turns"]
    ].fillna(0)

    monthly["web_query_rate"] = monthly["web_query_turns"] / monthly["total_turns"]
    monthly["thought_rate"] = monthly["thought_turns"] / monthly["total_turns"]
    # print(monthly["web_query_turns"].sum(), monthly["thought_turns"].sum())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["web_query_rate"],
            mode="lines+markers",
            name="Turns with Web Queries",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["thought_rate"],
            mode="lines+markers",
            name="Turns with Thoughts",
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Turns (%)",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%")

    file_name = "distribution_of_web_query_and_thoughts_over_time"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def count_models_with_web_queries():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    ).copy()

    if "openai_models" not in df.columns:
        print("Column `openai_models` not found.")
        return pd.DataFrame(), pd.DataFrame()

    def _as_model_list(value):
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str) and v.strip()]
        if pd.isna(value):
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [v for v in parsed if isinstance(v, str) and v.strip()]
            except Exception:
                pass
            return [text]
        return []

    def _has_web_queries(value):
        if isinstance(value, list):
            parsed = value
        elif isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return False
        else:
            return False
        return any(isinstance(item, list) and len(item) > 0 for item in parsed)

    def _primary_model(models):
        cleaned = [model for model in models if isinstance(model, str) and model]
        if not cleaned:
            return "Unknown"
        return cleaned[-1]

    df["openai_models"] = df["openai_models"].apply(_as_model_list)
    df["has_web_queries"] = df["web_queries"].apply(_has_web_queries)
    df["primary_model"] = df["openai_models"].apply(_primary_model)

    model_coverage = (
        df.groupby("primary_model")
        .agg(
            total_usage=("primary_model", "size"),
            with_web_queries=("has_web_queries", "sum"),
        )
        .reset_index()
        .rename(columns={"primary_model": "model"})
    )
    model_coverage["without_web_queries"] = (
        model_coverage["total_usage"] - model_coverage["with_web_queries"]
    )
    model_coverage["web_query_coverage_rate"] = (
        model_coverage["with_web_queries"] / model_coverage["total_usage"]
    )
    model_coverage = model_coverage.sort_values(
        ["web_query_coverage_rate", "with_web_queries", "total_usage"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    print("\nPrimary-model coverage for web queries:")
    print(model_coverage.to_string(index=False))

    return model_coverage

#                       model  total_usage  with_web_queries  without_web_queries  web_query_coverage_rate
#           gpt-5-2-thinking            8                 8                    0                 1.000000
#              gpt-5-a-t-mini            4                 4                    0                 1.000000
#         gpt-5-auto-thinking            3                 3                    0                 1.000000
#                          o3          268               267                    1                 0.996269
#                gpt-5-t-mini          714               696                   18                 0.974790
#              gpt-5-thinking          475               462                   13                 0.972632
#                     gpt-5-1          431               418                   13                 0.969838
#                     o4-mini          326               313                   13                 0.960123
#                     gpt-5-2          663               611                   52                 0.921569
#                o4-mini-high           79                72                    7                 0.911392
#            gpt-5-1-thinking           54                41                   13                 0.759259
#                       gpt-5         9997              5587                 4410                 0.558868
#                      gpt-4o        23706              6916                16790                 0.291740
#                     gpt-4-1           38                 9                   29                 0.236842
#                     Unknown          717                86                  631                 0.119944
#                  gpt-5-mini         2381               187                 2194                 0.078538
#                     gpt-4-5           25                 1                   24                 0.040000
#                 gpt-4o-mini         1247                 0                 1247                 0.000000
#                gpt-4-1-mini          809                 0                  809                 0.000000
#                     o3-mini          131                 0                  131                 0.000000
#                o3-mini-high           14                 0                   14                 0.000000
# text-davinci-002-render-sha           11                 0                   11                 0.000000
#               gpt-5-instant            2                 0                    2                 0.000000
#                    research            1                 0                    1                 0.000000

def _normalize_reason_transition_endpoint(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, np.integer)):
        endpoint = str(int(value))
    elif isinstance(value, float):
        if value.is_integer():
            endpoint = str(int(value))
        else:
            endpoint = f"{value:.12g}"
    else:
        endpoint = str(value).strip()

    endpoint = endpoint.strip().strip("()")
    endpoint = endpoint.replace(" ", "")
    if not endpoint:
        return ""

    lowered = endpoint.lower()
    if lowered in {
        "u",
        "user",
        "userquery",
        "user_query",
        "userprompt",
        "user_prompt",
    }:
        return "U"
    return endpoint


def _normalize_query_reason_label(value):
    label = str(value or "").strip()
    lowered = label.lower()
    if not lowered:
        return ""
    if "other" in lowered:
        return "Other"
    if "hybrid" in lowered:
        return "Hybrid"
    if "expansion" in lowered:
        return "Query Expansion"
    if "rewriting" in lowered or "rewrite" in lowered:
        return "Query Rewriting"
    return label


def reasons_for_another_web_query():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )

    df = _filter_query_reformulation_df_for_reason(df).copy()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_name = "gpt-4o-mini"

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        all_thoughts = _safe_json_value(row.get("thoughts_list", []), [])
        all_web_queries = _safe_json_value(row.get("web_queries", []), [])
        user_query = _row_latest_user_query(row)
        if not user_query:
            continue

        structured_web_queries = []
        for loop_idx, query_group in enumerate(all_web_queries, start=1):
            if not isinstance(query_group, list):
                query_group = [query_group]
            cleaned_queries = [
                str(query).strip()
                for query in query_group
                if isinstance(query, str) and str(query).strip()
            ]
            if cleaned_queries:
                structured_web_queries.append(
                    {
                        "loop_idx": loop_idx,
                        "queries": cleaned_queries,
                    }
                )
        if not structured_web_queries:
            continue

        loop_query_records = {}
        structured_query_records = []
        for loop_entry in structured_web_queries:
            loop_idx = loop_entry["loop_idx"]
            loop_records = []
            for query_idx, query in enumerate(loop_entry["queries"], start=1):
                record = {
                    "query_id": f"{loop_idx}.{query_idx}",
                    "loop_idx": loop_idx,
                    "query_idx": query_idx,
                    "query": query,
                }
                loop_records.append(record)
                structured_query_records.append(record)
            loop_query_records[loop_idx] = loop_records

        web_queries_text = "\n".join(
            f"({item['query_id']}) {item['query']}"
            for item in structured_query_records
        )

        structured_thinking_records = []
        for loop_entry in structured_web_queries:
            loop_idx = loop_entry["loop_idx"]
            query_group = loop_entry["queries"]
            thought_group = (
                all_thoughts[loop_idx - 1]
                if isinstance(all_thoughts, list) and loop_idx - 1 < len(all_thoughts)
                else []
            )
            if isinstance(thought_group, list):
                joined_thought = " ".join(
                    str(thought).strip()
                    for thought in thought_group
                    if isinstance(thought, str) and str(thought).strip()
                ).strip()
            else:
                joined_thought = str(thought_group or "").strip()
            for query_idx, _query in enumerate(query_group, start=1):
                structured_thinking_records.append(
                    {
                        "query_id": f"{loop_idx}.{query_idx}",
                        "thinking_trace": joined_thought,
                    }
                )

        transition_candidates = []
        first_loop_idx = structured_web_queries[0]["loop_idx"]
        for to_record in loop_query_records.get(first_loop_idx, []):
            transition_candidates.append(
                {
                    "from": "U",
                    "to": to_record["query_id"],
                    "from_query": user_query,
                    "to_query": to_record["query"],
                    "from_loop_idx": 0,
                    "to_loop_idx": first_loop_idx,
                    "transition_kind": "user_to_first_web_turn",
                }
            )

        for loop_pos in range(len(structured_web_queries) - 1):
            from_loop_idx = structured_web_queries[loop_pos]["loop_idx"]
            to_loop_idx = structured_web_queries[loop_pos + 1]["loop_idx"]
            from_records = loop_query_records.get(from_loop_idx, [])
            to_records = loop_query_records.get(to_loop_idx, [])
            for from_record in from_records:
                for to_record in to_records:
                    transition_candidates.append(
                        {
                            "from": from_record["query_id"],
                            "to": to_record["query_id"],
                            "from_query": from_record["query"],
                            "to_query": to_record["query"],
                            "from_loop_idx": from_loop_idx,
                            "to_loop_idx": to_loop_idx,
                            "transition_kind": "web_turn_to_web_turn",
                        }
                    )

        if not transition_candidates:
            continue

        transition_candidates_text = "\n".join(
            f"({item['from']} -> {item['to']})"
            for item in transition_candidates
        )

        thinking_trace_lines = [
            f"({item['query_id']}) {item['thinking_trace']}"
            for item in structured_thinking_records
            if item["thinking_trace"]
        ]
        thinking_traces_text = "\n".join(thinking_trace_lines)

        try:
            reason_eval = _run_judge(
                client=client,
                model_name=model_name,
                system_prompt=SYSTEM_PROMPT_QUERY_REASON,
                user_prompt=USER_PROMPT_QUERY_REASON.format(
                    user_query=user_query,
                    web_queries=web_queries_text,
                    thinking_traces=thinking_traces_text,
                    transition_candidates=transition_candidates_text,
                ),
            )
        except Exception as e:
            print("reasons_for_another_web_query reason", row.get("conv_id"), row.get("turn_id"), e)
            continue

        reason_parsed = reason_eval["parsed_judgment"]
        transitions = (
            reason_parsed.get("transitions", [])
            if isinstance(reason_parsed, dict)
            else []
        )
        valid_transition_pairs = {
            (item["from"], item["to"]) for item in transition_candidates
        }
        reason_transition_by_pair = {}
        normalized_reason_transitions = []
        for transition in transitions:
            if not isinstance(transition, dict):
                continue
            from_id = _normalize_reason_transition_endpoint(transition.get("from"))
            to_id = _normalize_reason_transition_endpoint(transition.get("to"))
            pair = (from_id, to_id)
            if (
                not from_id
                or not to_id
                or pair not in valid_transition_pairs
                or pair in reason_transition_by_pair
            ):
                continue
            normalized_transition = {
                "from": from_id,
                "to": to_id,
                "label": _normalize_query_reason_label(transition.get("label", "")),
                "reasoning": str(transition.get("reasoning", "")).strip(),
            }
            reason_transition_by_pair[pair] = normalized_transition
            normalized_reason_transitions.append(normalized_transition)

        validation_transition_candidates = []
        for transition in transition_candidates:
            reversed_transition = {
                "from": transition["to"],
                "to": transition["from"],
                "from_query": transition["to_query"],
                "to_query": transition["from_query"],
                "from_loop_idx": transition["to_loop_idx"],
                "to_loop_idx": transition["from_loop_idx"],
                "transition_kind": transition["transition_kind"],
            }
            validation_transition_candidates.append(reversed_transition)

        validation_transition_candidates_text = "\n".join(
            f"({item['from']} -> {item['to']})"
            for item in validation_transition_candidates
        )
        valid_validation_transition_pairs = {
            (item["from"], item["to"]) for item in validation_transition_candidates
        }

        try:
            validation_eval = _run_judge(
                client=client,
                model_name=model_name,
                system_prompt=SYSTEM_PROMPT_QUERY_REASON_VALIDATOR,
                user_prompt=USER_PROMPT_QUERY_REASON_VALIDATOR.format(
                    user_query=user_query,
                    web_queries=web_queries_text,
                    transition_candidates=validation_transition_candidates_text,
                ),
            )
        except Exception as e:
            print("reasons_for_another_web_query validate", row.get("conv_id"), row.get("turn_id"), e)
            continue

        validator_parsed = validation_eval["parsed_judgment"]
        validator_transitions = (
            validator_parsed.get("transitions", [])
            if isinstance(validator_parsed, dict)
            else []
        )
        validator_transition_by_pair = {}
        normalized_validator_transitions = []
        for transition in validator_transitions:
            if not isinstance(transition, dict):
                continue
            from_id = _normalize_reason_transition_endpoint(transition.get("from"))
            to_id = _normalize_reason_transition_endpoint(transition.get("to"))
            pair = (from_id, to_id)
            if (
                not from_id
                or not to_id
                or pair not in valid_validation_transition_pairs
                or pair in validator_transition_by_pair
            ):
                continue

            normalized_transition = {
                "from": from_id,
                "to": to_id,
                "label": _normalize_query_reason_label(transition.get("label", "")),
                "reasoning": str(transition.get("reasoning", "")).strip(),
            }
            validator_transition_by_pair[pair] = normalized_transition
            normalized_validator_transitions.append(normalized_transition)

        records.append(
            {
                "user_id": row.get("user_id"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "topic": row.get("topic"),
                "language": row.get("language"),
                "user_query": user_query,
                "web_queries": structured_query_records,
                "web_queries_structured_text": web_queries_text,
                "thinking_traces": structured_thinking_records,
                "transition_candidates": transition_candidates,
                "transition_candidates_text": transition_candidates_text,
                "validator_transition_candidates": validation_transition_candidates,
                "validator_transition_candidates_text": validation_transition_candidates_text,
                "query_reason_parsed_judgment_judgment": reason_eval["parsed_judgment"],
                "query_reason_validator_parsed_judgment_judgment": validation_eval["parsed_judgment"],
                "query_reason_transitions_normalized": normalized_reason_transitions,
                "query_reason_validator_transitions_normalized": normalized_validator_transitions,
            }
        )

    records_df = pd.DataFrame(records)
    records_df.to_csv(
        f"{OUTPUT_PATH}/metadata/query_reformulations_web_query_transition_reasons.csv",
        index=False,
    )
    records_df.to_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulations_web_query_transition_reasons.pkl"
    )
    to_json(
        records,
        f"{OUTPUT_PATH}/metadata/query_reformulations_web_query_transition_reasons.json",
    )
    return records_df


def plot_reasons_for_another_web_query_distribution():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulations_web_query_transition_reasons.pkl"
    ).copy()

    if df.empty:
        print("No query-transition reason data found.")
        return pd.DataFrame()

    iteration_totals_before = Counter()
    iteration_reason_counts_before = Counter()
    query_iteration_totals_before = Counter()
    query_iteration_reason_counts_before = Counter()
    iteration_totals_after = Counter()
    iteration_reason_counts_after = Counter()
    other_before_total = 0
    other_before_count = 0
    other_after_total = 0
    other_after_count = 0

    def _as_dict(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(value)
                except Exception:
                    return {}
        return {}

    def _coerce_int(value):
        if value is None:
            return None
        try:
            if isinstance(value, float) and pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_loop_idx_from_endpoint(endpoint):
        normalized = _normalize_reason_transition_endpoint(endpoint)
        if not normalized or normalized == "U":
            return None
        try:
            return int(str(normalized).split(".", 1)[0])
        except (TypeError, ValueError):
            return None

    reason_order = [
        "Query Rewriting",
        "Query Expansion",
        "Hybrid",
        "Other"
    ]
    aggregate_from_iteration = 3

    def _bucket_iteration(iteration_idx):
        if iteration_idx >= aggregate_from_iteration:
            return aggregate_from_iteration
        return iteration_idx

    def _transition_group_label(iteration_idx, use_arrow=False):
        arrow = " → " if use_arrow else " -> "
        if iteration_idx == 1:
            return f"User{arrow}Iter. 1"
        if iteration_idx >= aggregate_from_iteration:
            return f"Iter. {aggregate_from_iteration-1}+{arrow}Iter. {aggregate_from_iteration}+"
        return f"Iter. {iteration_idx - 1}{arrow}Iter. {iteration_idx}"

    def _build_transition_meta_by_pair(transitions):
        transition_meta = {}
        for transition in transitions:
            if not isinstance(transition, dict):
                continue
            transition_key = (
                _normalize_reason_transition_endpoint(transition.get("from")),
                _normalize_reason_transition_endpoint(transition.get("to")),
            )
            if not transition_key[0] or not transition_key[1]:
                continue
            transition_meta[transition_key] = {
                "transition_kind": str(
                    transition.get("transition_kind", "")
                ).strip(),
                "from_loop_idx": _coerce_int(transition.get("from_loop_idx")),
                "to_loop_idx": _coerce_int(transition.get("to_loop_idx")),
            }
        return transition_meta

    def _infer_iteration_idx(transition_key, transition_meta, flipped=False):
        transition_kind = str(transition_meta.get("transition_kind", "")).strip()
        from_loop_idx = transition_meta.get("from_loop_idx")
        to_loop_idx = transition_meta.get("to_loop_idx")

        if from_loop_idx is None:
            from_loop_idx = _parse_loop_idx_from_endpoint(transition_key[0])
        if to_loop_idx is None:
            to_loop_idx = _parse_loop_idx_from_endpoint(transition_key[1])

        if flipped:
            if transition_key[1] == "U" or transition_kind == "user_to_first_web_turn":
                return 1
            if (
                from_loop_idx is not None
                and to_loop_idx is not None
                and from_loop_idx == to_loop_idx + 1
            ):
                return from_loop_idx
            return None

        if transition_key[0] == "U" or transition_kind == "user_to_first_web_turn":
            return 1
        if (
            from_loop_idx is not None
            and to_loop_idx is not None
            and to_loop_idx == from_loop_idx + 1
        ):
            return to_loop_idx
        return None

    def _aggregate_labels_for_destination_query(labels):
        normalized_labels = [label for label in labels if label in reason_order]
        if not normalized_labels:
            return ""

        label_set = set(normalized_labels)
        if "Hybrid" in label_set:
            return "Hybrid"
        if "Query Rewriting" in label_set and "Query Expansion" in label_set:
            return "Hybrid"

        non_other_labels = label_set - {"Other"}
        if non_other_labels == {"Query Rewriting"}:
            return "Query Rewriting"
        if non_other_labels == {"Query Expansion"}:
            return "Query Expansion"
        if not non_other_labels:
            return "Other"

        if len(non_other_labels) == 1:
            return next(iter(non_other_labels))
        return "Hybrid"

    def _build_plot_df(iteration_totals, iteration_reason_counts):
        if not iteration_totals:
            return pd.DataFrame()

        iteration_values = sorted(iteration_totals.keys())
        rows = []
        for iteration_idx in iteration_values:
            total = iteration_totals[iteration_idx]
            for reason in reason_order:
                count = iteration_reason_counts.get((iteration_idx, reason), 0)
                rate = (count / total) if total > 0 else 0.0
                rows.append(
                    {
                        "iteration": iteration_idx,
                        "reason": reason,
                        "count": count,
                        "total": total,
                        "rate": rate,
                        "transition_group": _transition_group_label(
                            iteration_idx,
                            use_arrow=False,
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def _save_reason_rate_plot(plot_df, file_name):
        if len(plot_df) == 0:
            return

        color_map = {
            "Query Rewriting": "#1f77b4",
            "Query Expansion": "#2ca02c",
            "Hybrid": "#ff7f0e",
            "Other": "#f3ff0e"
        }
        fig = go.Figure()
        iteration_values = sorted(plot_df["iteration"].unique())

        for reason in reason_order:
            reason_df = plot_df[plot_df["reason"] == reason].sort_values("iteration")
            if len(reason_df) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=reason_df["iteration"],
                    y=reason_df["rate"],
                    mode="lines+markers",
                    name=reason,
                    line=dict(width=3, color=color_map.get(reason)),
                    marker=dict(size=9),
                    customdata=reason_df[["count", "total"]].values,
                    hovertemplate=(
                        "Reason: %{fullData.name}<br>"
                        "Iteration: %{x}<br>"
                        "Rate: %{y:.1%}<br>"
                        "Count: %{customdata[0]} / %{customdata[1]}"
                        "<extra></extra>"
                    ),
                )
            )

        ticktext = [
            _transition_group_label(iteration_idx, use_arrow=True)
            for iteration_idx in iteration_values
        ]

        y_max = float(plot_df["rate"].max()) if len(plot_df) > 0 else 0.0
        y_upper = min(1.0, max(0.05, y_max * 1.15))
        fig.update_layout(
            # title="Reason Rates Over Query Reformulation Iteration",
            # width=980,
            # height=650,
            margin=dict(t=80, b=80, l=80, r=110),
            xaxis_title="Web Query Iteration",
            yaxis_title="Rate",
            legend_title="",
        )
        x_min = min(iteration_values)
        x_max = max(iteration_values)
        fig.update_xaxes(
            tickmode="array",
            tickvals=iteration_values,
            ticktext=ticktext,
            automargin=True,
            range=[x_min - 0.2, x_max + 0.5],
        )
        fig.update_yaxes(tickformat=".0%", range=[-0.05, y_upper+0.05])

        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(20, 20), legend_pos=(0.9, 1.2))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    for _, row in df.iterrows():
        normalized_reason_transitions = _safe_json_value(
            row.get("query_reason_transitions_normalized", []),
            [],
        )
        normalized_validator_transitions = _safe_json_value(
            row.get("query_reason_validator_transitions_normalized", []),
            [],
        )
        transition_candidates = _safe_json_value(
            row.get("transition_candidates", []),
            [],
        )
        validator_transition_candidates = _safe_json_value(
            row.get("validator_transition_candidates", []),
            [],
        )

        if not isinstance(normalized_reason_transitions, list):
            normalized_reason_transitions = []
        if not isinstance(normalized_validator_transitions, list):
            normalized_validator_transitions = []
        if not isinstance(transition_candidates, list):
            transition_candidates = []
        if not isinstance(validator_transition_candidates, list):
            validator_transition_candidates = []

        if not validator_transition_candidates and transition_candidates:
            for transition in transition_candidates:
                if not isinstance(transition, dict):
                    continue
                validator_transition_candidates.append(
                    {
                        "from": transition.get("to"),
                        "to": transition.get("from"),
                        "from_loop_idx": transition.get("to_loop_idx"),
                        "to_loop_idx": transition.get("from_loop_idx"),
                        "transition_kind": transition.get("transition_kind", ""),
                    }
                )

        reason_judgment = _as_dict(
            row.get("query_reason_parsed_judgment_judgment", {})
        )
        validator_judgment = _as_dict(
            row.get("query_reason_validator_parsed_judgment_judgment", {})
        )

        original_transitions = normalized_reason_transitions
        if not original_transitions:
            original_transitions = (
                reason_judgment.get("transitions", [])
                if isinstance(reason_judgment, dict)
                else []
            )
        validator_transitions = normalized_validator_transitions
        if not validator_transitions:
            validator_transitions = (
                validator_judgment.get("transitions", [])
                if isinstance(validator_judgment, dict)
                else []
            )

        validator_by_pair = {}
        for transition in validator_transitions:
            if not isinstance(transition, dict):
                continue
            transition_key = (
                _normalize_reason_transition_endpoint(transition.get("from")),
                _normalize_reason_transition_endpoint(transition.get("to")),
            )
            if not transition_key[0] or not transition_key[1]:
                continue
            validator_by_pair[transition_key] = {
                "label": _normalize_query_reason_label(transition.get("label", "")),
                "reasoning": str(transition.get("reasoning", "")).strip(),
            }

        reason_by_pair = {}
        for transition in original_transitions:
            if not isinstance(transition, dict):
                continue
            transition_key = (
                _normalize_reason_transition_endpoint(transition.get("from")),
                _normalize_reason_transition_endpoint(transition.get("to")),
            )
            if not transition_key[0] or not transition_key[1]:
                continue
            reason_by_pair[transition_key] = transition

        transition_meta_by_pair = _build_transition_meta_by_pair(transition_candidates)
        validator_transition_meta_by_pair = _build_transition_meta_by_pair(
            validator_transition_candidates
        )

        transition_keys_before = list(transition_meta_by_pair.keys())
        if not transition_keys_before:
            transition_keys_before = list(reason_by_pair.keys())

        incoming_labels_by_destination_query = {}
        destination_query_iteration_bucket = {}

        for transition_key in transition_keys_before:
            original_label = _normalize_query_reason_label(
                reason_by_pair.get(transition_key, {}).get("label", "")
            )
            other_before_total += 1
            if original_label == "Other":
                other_before_count += 1

            iteration_idx = _infer_iteration_idx(
                transition_key,
                transition_meta_by_pair.get(transition_key, {}),
                flipped=False,
            )
            if iteration_idx is None:
                continue

            iteration_bucket = _bucket_iteration(iteration_idx)
            iteration_totals_before[iteration_bucket] += 1
            if original_label in reason_order:
                iteration_reason_counts_before[(iteration_bucket, original_label)] += 1

                destination_query = transition_key[1]
                if destination_query:
                    incoming_labels_by_destination_query.setdefault(
                        destination_query,
                        [],
                    ).append(original_label)
                    destination_query_iteration_bucket.setdefault(
                        destination_query,
                        iteration_bucket,
                    )

        for destination_query, incoming_labels in incoming_labels_by_destination_query.items():
            aggregate_label = _aggregate_labels_for_destination_query(incoming_labels)
            if aggregate_label not in reason_order:
                continue

            iteration_bucket = destination_query_iteration_bucket.get(destination_query)
            if iteration_bucket is None:
                continue

            query_iteration_totals_before[iteration_bucket] += 1
            query_iteration_reason_counts_before[(iteration_bucket, aggregate_label)] += 1

        transition_keys_after = list(validator_transition_meta_by_pair.keys())
        if not transition_keys_after:
            transition_keys_after = list(validator_by_pair.keys())

        for transition_key in transition_keys_after:
            validator_label = _normalize_query_reason_label(
                validator_by_pair.get(transition_key, {}).get("label", "")
            )
            other_after_total += 1
            if validator_label == "Other":
                other_after_count += 1

            iteration_idx = _infer_iteration_idx(
                transition_key,
                validator_transition_meta_by_pair.get(transition_key, {}),
                flipped=True,
            )
            if iteration_idx is None:
                continue

            iteration_bucket = _bucket_iteration(iteration_idx)
            iteration_totals_after[iteration_bucket] += 1
            if validator_label in reason_order:
                iteration_reason_counts_after[(iteration_bucket, validator_label)] += 1

    if not iteration_totals_before and not iteration_totals_after:
        print("No transitions matched the requested iteration definition for plotting.")
        return pd.DataFrame()

    plot_df_before = _build_plot_df(iteration_totals_before, iteration_reason_counts_before)
    query_plot_df_before = _build_plot_df(
        query_iteration_totals_before,
        query_iteration_reason_counts_before,
    )
    plot_df_after = _build_plot_df(iteration_totals_after, iteration_reason_counts_after)

    if (
        len(plot_df_before) == 0
        and len(query_plot_df_before) == 0
        and len(plot_df_after) == 0
    ):
        print("No iteration-rate rows to plot.")
        return pd.DataFrame()

    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    plot_df_before.to_csv(
        f"{OUTPUT_PATH}/metadata/query_reformulations_web_query_transition_reason_rates_by_iteration.csv",
        index=False,
    )
    if len(query_plot_df_before) > 0:
        query_plot_df_before.to_csv(
            f"{OUTPUT_PATH}/metadata/query_reformulations_web_query_reason_rates_by_iteration_query_level_before_validation.csv",
            index=False,
        )
    if len(plot_df_after) > 0:
        plot_df_after.to_csv(
            f"{OUTPUT_PATH}/metadata/query_reformulations_web_query_transition_reason_rates_by_iteration_after_validation.csv",
            index=False,
        )

    _save_reason_rate_plot(
        query_plot_df_before,
        file_name="reasons_for_another_web_query_distribution",
    )
    _save_reason_rate_plot(
        query_plot_df_before,
        file_name="reasons_for_another_web_query_distribution_query_level_before_validation",
    )
    _save_reason_rate_plot(
        plot_df_before,
        file_name="reasons_for_another_web_query_distribution_transition_level_before_validation",
    )
    _save_reason_rate_plot(
        plot_df_after,
        file_name="reasons_for_another_web_query_distribution_after_validation",
    )

    if len(plot_df_before) > 0:
        print("\nReason rates by iteration (before validation):")
        print(plot_df_before.to_string(index=False))
    if len(plot_df_after) > 0:
        print("\nReason rates by iteration (after validation):")
        print(plot_df_after.to_string(index=False))
    if len(query_plot_df_before) > 0:
        print("\nReason rates by iteration (query-level before validation):")
        print(query_plot_df_before.to_string(index=False))

    query_other_before_total = int(sum(query_iteration_totals_before.values()))
    query_other_before_count = int(
        sum(
            query_iteration_reason_counts_before.get((iteration_idx, "Other"), 0)
            for iteration_idx in query_iteration_totals_before.keys()
        )
    )

    before_rate = (
        (other_before_count / other_before_total) if other_before_total > 0 else 0.0
    )
    after_rate = (
        (other_after_count / other_after_total) if other_after_total > 0 else 0.0
    )
    print(
        f"\n'Other' before validation: {other_before_count}/{other_before_total} "
        f"({before_rate:.2%})."
    )
    print(
        f"'Other' after validation: {other_after_count}/{other_after_total} "
        f"({after_rate:.2%})."
    )
    if query_other_before_total > 0:
        print(
            f"'Other' before validation (query-level): "
            f"{query_other_before_count}/{query_other_before_total} "
            f"({query_other_before_count / query_other_before_total:.2%})."
        )

    return plot_df_before

def _parse_eval_json(text):
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                return {}
    return {}


def _normalize_query_relation_type(value):
    label = str(value or "").strip().lower()
    if label == "generative":
        return "discretionary"
    return label


def _run_judge(client, model_name, system_prompt, user_prompt):
    # Keep evaluator calls tool-free: this judge should only classify the
    # provided text and must not browse or invoke external tools.
    response = client.responses.create(
        model=model_name,
        tools=[],
        tool_choice="none",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )
    raw_text = response.output_text
    return {
        "raw_judgment": raw_text,
        "parsed_judgment": _parse_eval_json(raw_text),
    }


def _safe_json_value(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return default
    return value


def _dedupe_preserve_order(values):
    seen = set()
    deduped = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _sanitize_file_component(value):
    value = str(value or "").strip()
    safe_chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "unknown"


def _replay_output_prefix(replay_path, response_mode):
    replay_name = os.path.splitext(os.path.basename(replay_path))[0]
    return (
        f"replay_{_sanitize_file_component(replay_name)}_"
        f"{_sanitize_file_component(response_mode)}"
    )


def _message_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _extract_user_prompt_from_replay_row(row):
    user_prompt = row.get("user_prompt", "")
    if isinstance(user_prompt, str) and user_prompt.strip():
        return user_prompt

    prompt = row.get("prompt", [])
    if not isinstance(prompt, list):
        return ""

    for message in reversed(prompt):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return _message_content_to_text(message.get("content", ""))
    return ""


def _extract_web_query_groups_from_response(response_payload):
    if not isinstance(response_payload, dict):
        return []

    web_query_groups = []
    for item in response_payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "web_search_call":
            continue

        action = item.get("action") or {}
        queries = []

        action_queries = action.get("queries", [])
        if isinstance(action_queries, list):
            queries.extend(action_queries)

        queries = _dedupe_preserve_order(queries)
        if queries:
            web_query_groups.append(queries)

    return web_query_groups


def _extract_replay_web_query_groups(row, response_mode):
    mode_payload = row.get(response_mode, {})
    if not isinstance(mode_payload, dict):
        return []
    return _extract_web_query_groups_from_response(mode_payload.get("response", {}))


def _load_replay_query_reformulation_df(
    replay_path=f"{OUTPUT_PATH}/replays/gpt-5-mini-2025-08-07.json",
    response_mode="auto",
):
    replay_data = load_json(replay_path)
    if not isinstance(replay_data, dict):
        return pd.DataFrame()

    rows = []
    for result_key, row in replay_data.items():
        if not isinstance(row, dict):
            continue

        user_prompt = _extract_user_prompt_from_replay_row(row)
        web_query_groups = _extract_replay_web_query_groups(row, response_mode)
        rows.append(
            {
                "result_key": row.get("result_key", result_key),
                "sample_source": row.get("sample_source"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "topic": row.get("topic"),
                "language": row.get("language"),
                "invivo_model": row.get("invivo_model"),
                "replay_model": row.get("replay_model"),
                "response_mode": response_mode,
                "user_prompt": user_prompt,
                "user_msg_history": [user_prompt] if user_prompt else [],
                "assistant_msg_history": [],
                "web_queries": json.dumps(web_query_groups, ensure_ascii=False),
                "thoughts_list": json.dumps([[] for _ in web_query_groups]),
                "sources": json.dumps([[] for _ in web_query_groups]),
                "memories": json.dumps([[] for _ in web_query_groups]),
            }
        )

    return pd.DataFrame(rows)


def _flatten_web_query_groups(web_query_groups):
    web_query_groups = _safe_json_value(web_query_groups, [])
    web_queries = []
    for query_group in web_query_groups:
        if isinstance(query_group, list):
            web_queries.extend(query_group)
        else:
            web_queries.append(query_group)
    return [
        query.strip()
        for query in web_queries
        if isinstance(query, str) and query.strip()
    ]


def _first_web_query_group(web_query_groups):
    web_query_groups = _safe_json_value(web_query_groups, [])
    if not web_query_groups:
        return []
    first_group = web_query_groups[0]
    if isinstance(first_group, list):
        return [
            query.strip()
            for query in first_group
            if isinstance(query, str) and query.strip()
        ]
    return [first_group.strip()] if isinstance(first_group, str) and first_group.strip() else []


def _get_loop_items(loop_items, loop_idx):
    if not isinstance(loop_items, list) or loop_idx >= len(loop_items):
        return []
    items = loop_items[loop_idx]
    if isinstance(items, list):
        return items
    return [items]


def _detect_web_query_token_sources(
    latest_user_query,
    web_query_groups,
    previous_user_or_assistant_history=None,
    all_sources=None,
    all_memories=None,
):
    latest_user_query_words = set(preprocess_text_in_chunks(latest_user_query or ""))
    previous_history_words = set(
        preprocess_texts(previous_user_or_assistant_history or [])
    )

    web_query_groups = _safe_json_value(web_query_groups, [])
    all_sources = _safe_json_value(all_sources, [])
    all_memories = _safe_json_value(all_memories, [])

    new_words_from_user_queries = []
    new_words_from_assistant_queries = []
    new_words_from_thoughts = []
    new_words_from_sources = []
    new_words_from_memories = []
    new_words_from_pk = []

    for loop_idx, system_queries in enumerate(web_query_groups):
        if not isinstance(system_queries, list):
            system_queries = [system_queries]

        sources_w = set(preprocess_texts(_get_loop_items(all_sources, loop_idx)))
        memories_w = set(preprocess_texts(_get_loop_items(all_memories, loop_idx)))

        for system_query in system_queries:
            for word in preprocess_text_in_chunks(system_query):
                if word in latest_user_query_words:
                    new_words_from_user_queries.append(word)
                elif word in previous_history_words:
                    # Backward-compatible column name: this includes previous
                    # user messages as well as assistant messages.
                    new_words_from_assistant_queries.append(word)
                elif word in memories_w:
                    new_words_from_memories.append(word)
                elif word in sources_w:
                    new_words_from_sources.append(word)
                else:
                    new_words_from_pk.append(word)

    return {
        "all_new_words_from_user_queries": list(set(new_words_from_user_queries)),
        "all_new_words_from_assistant_queries": list(
            set(new_words_from_assistant_queries)
        ),
        "all_new_words_from_sources": list(set(new_words_from_sources)),
        "all_new_words_from_thoughts": list(set(new_words_from_thoughts)),
        "all_new_words_from_memories": list(set(new_words_from_memories)),
        "all_new_words_from_pk": list(set(new_words_from_pk)),
    }


def _add_web_query_token_source_columns(df):
    token_source_rows = {
        "all_new_words_from_user_queries": [],
        "all_new_words_from_assistant_queries": [],
        "all_new_words_from_sources": [],
        "all_new_words_from_thoughts": [],
        "all_new_words_from_memories": [],
        "all_new_words_from_pk": [],
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_msg_history = _safe_json_value(row.get("user_msg_history"), [])
        assistant_msg_history = _safe_json_value(row.get("assistant_msg_history"), [])
        if isinstance(user_msg_history, str):
            user_msg_history = [user_msg_history]
        if isinstance(assistant_msg_history, str):
            assistant_msg_history = [assistant_msg_history]

        latest_user_query = user_msg_history[-1] if user_msg_history else ""
        previous_history = user_msg_history[:-1] + assistant_msg_history

        token_sources = _detect_web_query_token_sources(
            latest_user_query=latest_user_query,
            web_query_groups=row.get("web_queries"),
            previous_user_or_assistant_history=previous_history,
            all_sources=row.get("sources"),
            all_memories=row.get("memories"),
        )
        for col in token_source_rows:
            token_source_rows[col].append(token_sources[col])

    df = df.copy()
    for col, values in token_source_rows.items():
        df[col] = values
    return df


def _save_dataframe(df, output_stem):
    csv_path = f"{OUTPUT_PATH}/metadata/{output_stem}.csv"
    pkl_path = f"{OUTPUT_PATH}/metadata/{output_stem}.pkl"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)


def _filter_query_reformulation_df(df):
    filtered_rows = []
    num_wq = 0

    for _, row in df.iterrows():
        if row["conv_starter"] != 1:
            continue

        if row["language"] != "en":
            continue

        user_msg_history = row["user_msg_history"]
        if len(user_msg_history) != 1:
            continue

        user_query = user_msg_history[-1]
        if not user_query:
            continue

        web_queries = json.loads(row["web_queries"])
        all_web_queries = [q for qs in web_queries for q in qs]
        num_wq += len(all_web_queries)

        # if not all_web_queries:
        #     continue

        filtered_rows.append(row.to_dict())

    print(len(filtered_rows), num_wq)
    return pd.DataFrame(filtered_rows)


def _filter_query_reformulation_df_for_relation(df):
    filtered_rows = []

    for _, row in df.iterrows():
        if row["conv_starter"] != 1:
            continue

        if row["language"] != "en":
            continue

        user_msg_history = row["user_msg_history"]
        if len(user_msg_history) != 1:
            continue

        user_query = user_msg_history[-1]
        if not user_query:
            continue

        web_queries = json.loads(row["web_queries"])
        if not web_queries:
            continue

        web_queries = web_queries[0]
        if not web_queries:
            continue

        filtered_rows.append(row.to_dict())

    print(len(filtered_rows))
    return pd.DataFrame(filtered_rows)



def _filter_query_reformulation_df_for_reason(df):
    filtered_rows = []

    for _, row in df.iterrows():
        if row["conv_starter"] != 1:
            continue

        if row["language"] != "en":
            continue

        user_msg_history = _safe_json_value(row.get("user_msg_history"), [])
        if isinstance(user_msg_history, str):
            user_msg_history = [user_msg_history]
        if len(user_msg_history) != 1:
            continue

        user_query = user_msg_history[-1]
        if not user_query:
            continue

        web_queries = _safe_json_value(row.get("web_queries"), [])
        if not isinstance(web_queries, list) or not web_queries:
            continue
        has_non_empty_loop = any(
            isinstance(query_group, list)
            and any(
                isinstance(query, str) and query.strip()
                for query in query_group
            )
            for query_group in web_queries
        )
        if not has_non_empty_loop:
            continue

        filtered_rows.append(row.to_dict())

    print(len(filtered_rows))
    return pd.DataFrame(filtered_rows)


def _row_latest_user_query(row):
    user_msg_history = _safe_json_value(row.get("user_msg_history"), [])
    if isinstance(user_msg_history, str):
        user_msg_history = [user_msg_history]
    if user_msg_history:
        return user_msg_history[-1]
    return row.get("user_prompt") or row.get("user_query") or ""


def _base_query_record(row):
    return {
        "result_key": row.get("result_key"),
        "sample_source": row.get("sample_source"),
        "user_id": row.get("user_id"),
        "conv_id": row.get("conv_id"),
        "turn_id": row.get("turn_id"),
        "topic": row.get("topic"),
        "language": row.get("language"),
        "invivo_model": row.get("invivo_model"),
        "replay_model": row.get("replay_model"),
        "response_mode": row.get("response_mode"),
    }


def _save_query_eval_records(records, output_stem):
    results_df = pd.DataFrame(records)
    _save_dataframe(results_df, output_stem)
    to_json(records, f"{OUTPUT_PATH}/metadata/{output_stem}.json")
    return results_df


def _evaluate_type_of_user_and_web_query_df(
    df,
    output_stem,
    model_name="gpt-4o-mini",
    client=None,
):
    client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    num_wq = 0

    records = []
    print(f"Loaded {len(df)} rows")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_query = _row_latest_user_query(row)
        if not user_query:
            continue
        web_queries = _flatten_web_query_groups(row.get("web_queries"))
        num_wq += len(web_queries)

        try:
            eval_result = _run_judge(
                client=client,
                model_name=model_name,
                system_prompt=SYSTEM_PROMPT_USER_WEB_QUERY_TYPE,
                user_prompt=USER_PROMPT_USER_WEB_QUERY_TYPE.format(user_query=user_query),
            )
        except Exception as e:
            print("type_of_user_query", row.get("conv_id"), row.get("turn_id"), e)
            continue

        parsed = eval_result["parsed_judgment"]
        row_record = _base_query_record(row)
        row_record.update(
            {
                "user_query": user_query,
                "query_type": parsed.get("type"),
                "query_type_reasoning": parsed.get("reasoning"),
                "web_queries": json.dumps(web_queries, ensure_ascii=False),
            }
        )

        web_query_type_info = {}

        for wq in web_queries:
            try:
                eval_result = _run_judge(
                    client=client,
                    model_name=model_name,
                    system_prompt=SYSTEM_PROMPT_USER_WEB_QUERY_TYPE,
                    user_prompt=USER_PROMPT_USER_WEB_QUERY_TYPE.format(user_query=wq),
                )
            except Exception as e:
                print("type_of_web_query", row.get("conv_id"), row.get("turn_id"), e)
                continue

            parsed = eval_result["parsed_judgment"]
            web_query_type_info[wq] = {
                "query_type": parsed.get("type"),
                "query_type_reasoning": parsed.get("reasoning"),
            }

        row_record["web_query_type_info"] = json.dumps(
            web_query_type_info, ensure_ascii=False
        )
        records.append(row_record)

    print(num_wq)
    return _save_query_eval_records(records, output_stem)


def type_of_user_and_web_query():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )
    df = _filter_query_reformulation_df(df)
    return _evaluate_type_of_user_and_web_query_df(
        df,
        output_stem="query_reformulations_user_query_types",
    )


def _evaluate_relation_of_user_to_web_query_df(
    df,
    output_stem,
    model_name="gpt-4o-mini",
    client=None,
):
    client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    records = []
    print(f"Loaded {len(df)} rows")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_query = _row_latest_user_query(row)
        web_queries = _first_web_query_group(row.get("web_queries"))
        if not user_query or not web_queries:
            continue

        try:
            eval_result = _run_judge(
                client=client,
                model_name=model_name,
                system_prompt=SYSTEM_PROMPT_QUERY_RELATION,
                user_prompt=USER_PROMPT_QUERY_RELATION.format(
                    user_query=user_query,
                    web_queries=json.dumps(web_queries, ensure_ascii=False, indent=2),
                ),
            )
        except Exception as e:
            print(
                "relation_of_user_to_web_query",
                row.get("conv_id"),
                row.get("turn_id"),
                e,
            )
            continue

        parsed = eval_result["parsed_judgment"]
        if not isinstance(parsed, dict):
            parsed = {}

        web_query_relation_info = {}
        for web_query in web_queries:
            judgment = parsed.get(web_query, {})
            if not isinstance(judgment, dict):
                judgment = {}

            web_query_relation_info[web_query] = {
                "query_relation_type": _normalize_query_relation_type(
                    judgment.get("type")
                ),
                "query_relation_reasoning": judgment.get("reasoning"),
            }

        row_record = _base_query_record(row)
        row_record.update(
            {
                "user_query": user_query,
                "web_queries": json.dumps(web_queries, ensure_ascii=False),
                "web_query_relation_info": json.dumps(
                    web_query_relation_info, ensure_ascii=False
                ),
                # "query_relation_raw_judgment": eval_result["raw_judgment"],
            }
        )
        records.append(row_record)

    return _save_query_eval_records(records, output_stem)


def relation_of_user_to_web_query():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )
    df = _filter_query_reformulation_df_for_relation(df)
    return _evaluate_relation_of_user_to_web_query_df(
        df,
        output_stem="query_reformulations_user_web_query_relations",
    )


def _plot_user_and_web_query_type_distribution_from_df(
    df,
):
    max_web_stage_bucket = 3

    query_type_order = [
        "navigational",
        "informational",
        "commercial",
        "transactional",
    ]
    query_type_ticktext = [
        "Navigational",
        "Informational",
        "Commercial",
        "Transactional",
    ]
    query_type_label_map = dict(zip(query_type_order, query_type_ticktext))

    query_type_counts = (
        df["query_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda x: x != ""]
        .value_counts()
        .reindex(query_type_order, fill_value=0)
        .rename_axis("query_type")
        .reset_index(name="count")
    )
    query_type_counts["query_type_display"] = query_type_counts["query_type"].map(
        query_type_label_map
    )

    web_query_type_counts = Counter()
    for _, row in df.iterrows():
        value = row.get("web_query_type_info", "{}")
        try:
            web_query_type_info = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            web_query_type_info = {}

        if not isinstance(web_query_type_info, dict):
            continue

        for judgment in web_query_type_info.values():
            if not isinstance(judgment, dict):
                continue
            web_query_type = (
                str(judgment.get("query_type", ""))
                .strip()
                .lower()
            )
            if web_query_type in query_type_label_map:
                web_query_type_counts[web_query_type] += 1

    web_query_type_counts = (
        pd.DataFrame(
            {
                "query_type": query_type_order,
                "count": [web_query_type_counts.get(qt, 0) for qt in query_type_order],
            }
        )
        .assign(
            query_type_display=lambda x: x["query_type"].map(query_type_label_map)
        )
    )

    # Build per-loop distributions:
    # stage 0 => user query, stages 1..N => web-query loop indices.
    stage_type_counts = {}

    def _bucket_stage_idx(stage_idx):
        if stage_idx <= 0:
            return 0
        return min(stage_idx, max_web_stage_bucket)

    def _add_stage_type_count(stage_idx, query_type):
        if query_type not in query_type_label_map:
            return
        stage_idx = _bucket_stage_idx(stage_idx)
        if stage_idx not in stage_type_counts:
            stage_type_counts[stage_idx] = Counter()
        stage_type_counts[stage_idx][query_type] += 1

    for _, row in df.iterrows():
        user_query_type = str(row.get("query_type", "")).strip().lower()
        _add_stage_type_count(0, user_query_type)

        value = row.get("web_query_type_info", "{}")
        try:
            web_query_type_info = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            web_query_type_info = {}

        if not isinstance(web_query_type_info, dict):
            continue

        query_type_by_query = {}
        for query_text, judgment in web_query_type_info.items():
            if not isinstance(query_text, str) or not isinstance(judgment, dict):
                continue
            web_query_type = str(judgment.get("query_type", "")).strip().lower()
            if web_query_type in query_type_label_map:
                query_type_by_query[query_text.strip()] = web_query_type

        web_query_groups_value = row.get("web_query_groups", "[]")
        try:
            web_query_groups = json.loads(web_query_groups_value)
        except (TypeError, json.JSONDecodeError):
            web_query_groups = []

        added_from_loops = False
        if isinstance(web_query_groups, list) and len(web_query_groups) > 0:
            for loop_idx, query_group in enumerate(web_query_groups, start=1):
                if not isinstance(query_group, list):
                    query_group = [query_group]
                for web_query in query_group:
                    if not isinstance(web_query, str):
                        continue
                    web_query_type = query_type_by_query.get(web_query.strip())
                    if web_query_type in query_type_label_map:
                        _add_stage_type_count(loop_idx, web_query_type)
                        added_from_loops = True

        # Fallback for rows where grouped loop metadata is unavailable.
        if not added_from_loops:
            web_queries_value = row.get("web_queries", "[]")
            try:
                web_queries = json.loads(web_queries_value)
            except (TypeError, json.JSONDecodeError):
                web_queries = []

            if not isinstance(web_queries, list):
                web_queries = []

            # Without loop groups, use a single fallback bucket for web queries.
            for web_query in web_queries:
                if not isinstance(web_query, str):
                    continue
                web_query_type = query_type_by_query.get(web_query.strip())
                if web_query_type in query_type_label_map:
                    _add_stage_type_count(1, web_query_type)

    stage_indices = sorted(stage_type_counts.keys())
    stage_labels = []
    for stage_idx in stage_indices:
        if stage_idx == 0:
            stage_labels.append("User")
        elif stage_idx == max_web_stage_bucket:
            stage_labels.append(f"Iter. {max_web_stage_bucket}+")
        else:
            stage_labels.append(f"Iter. {stage_idx}")

    stage_totals = {
        stage_idx: sum(stage_type_counts.get(stage_idx, Counter()).values())
        for stage_idx in stage_indices
    }

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=query_type_counts["query_type_display"],
            values=query_type_counts["count"],
            textinfo="label+percent",
            textposition="inside",
            automargin=True,
            hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            sort=False,
            showlegend=False,
        )
    )
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    file_name="user_query_type_distribution"
    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16), legend_pos=None)
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    if stage_indices:
        color_sequence = px.colors.qualitative.Plotly
        type_color_map = {
            query_type: color_sequence[idx % len(color_sequence)]
            for idx, query_type in enumerate(query_type_order)
        }
        fig = go.Figure()
        for query_type in query_type_order:
            rates = []
            counts = []
            totals = []
            for stage_idx in stage_indices:
                count = stage_type_counts.get(stage_idx, Counter()).get(query_type, 0)
                total = stage_totals.get(stage_idx, 0)
                rate = (count / total) if total > 0 else 0.0
                rates.append(rate)
                counts.append(count)
                totals.append(total)

            fig.add_trace(
                go.Bar(
                    x=stage_labels,
                    y=rates,
                    name=query_type_label_map[query_type],
                    marker_color=type_color_map[query_type],
                    customdata=np.column_stack([counts, totals]),
                    hovertemplate=(
                        "Stage: %{x}<br>"
                        "Type: %{fullData.name}<br>"
                        "Share: %{y:.1%}<br>"
                        "Count: %{customdata[0]} / %{customdata[1]}"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            barmode="stack",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="Query Formulation Iteration",
            yaxis_title="Share",
            legend_title="",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1.0])

        file_name = "user_and_web_query_type_distribution_by_iteration"
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(24, 24), legend_pos=(0.8, 1.25))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=web_query_type_counts["query_type_display"],
            values=web_query_type_counts["count"],
            textinfo="label+percent",
            textposition="inside",
            automargin=True,
            hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            sort=False,
            showlegend=False,
        )
    )
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    file_name="web_query_type_distribution"
    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def plot_user_and_web_query_type_distribution():
    user_type_pkl_path = (
        f"{OUTPUT_PATH}/metadata/query_reformulations_user_query_types.pkl"
    )
    user_type_csv_path = (
        f"{OUTPUT_PATH}/metadata/query_reformulations_user_query_types.csv"
    )
    if os.path.exists(user_type_pkl_path):
        df = pd.read_pickle(user_type_pkl_path).copy()
    else:
        df = pd.read_csv(user_type_csv_path).copy()

    source_df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )
    source_df = _filter_query_reformulation_df(source_df).copy()

    def _normalize_web_query_groups(value):
        web_query_groups = _safe_json_value(value, [])
        if not isinstance(web_query_groups, list):
            return []

        normalized_groups = []
        for query_group in web_query_groups:
            if not isinstance(query_group, list):
                query_group = [query_group]

            cleaned_queries = [
                str(query).strip()
                for query in query_group
                if isinstance(query, str) and str(query).strip()
            ]
            if cleaned_queries:
                normalized_groups.append(cleaned_queries)
        return normalized_groups

    source_df["user_query"] = source_df.apply(_row_latest_user_query, axis=1)
    source_df["web_query_groups_normalized"] = source_df["web_queries"].apply(
        _normalize_web_query_groups
    )

    group_lookup = {}
    for _, row in source_df.iterrows():
        key = (
            str(row.get("conv_id", "")),
            str(row.get("turn_id", "")),
            str(row.get("user_query", "")).strip(),
        )
        group_lookup.setdefault(key, []).append(
            row.get("web_query_groups_normalized", [])
        )

    def _row_web_query_groups(row):
        key = (
            str(row.get("conv_id", "")),
            str(row.get("turn_id", "")),
            str(row.get("user_query", "")).strip(),
        )
        candidates = group_lookup.get(key, [])
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates[0]

        flat_queries = _safe_json_value(row.get("web_queries"), [])
        if not isinstance(flat_queries, list):
            flat_queries = []
        flat_queries = [
            str(query).strip()
            for query in flat_queries
            if isinstance(query, str) and str(query).strip()
        ]

        for groups in candidates:
            candidate_flat = [query for group in groups for query in group]
            if candidate_flat == flat_queries:
                return groups

        return candidates[0]

    df["web_query_groups"] = df.apply(
        lambda row: json.dumps(_row_web_query_groups(row), ensure_ascii=False),
        axis=1,
    )
    _plot_user_and_web_query_type_distribution_from_df(df)


def _plot_user_web_query_relation_distribution_from_df(
    df,
    file_name="user_web_query_relation_distribution",
    venn_file_name="user_web_query_relation_venn",
    assignment_file_name="query_reformulations_user_web_query_relation_venn_assignments.csv",
):
    relation_order = [
        "extractive",
        "abstractive",
        "discretionary",
    ]
    relation_ticktext = [
        "Extractive",
        "Abstractive",
        "Discretionary",
    ]
    relation_label_map = dict(zip(relation_order, relation_ticktext))

    all_relations = []
    user_query_relation_rows = []
    for _, row in df.iterrows():
        value = row.get("web_query_relation_info", "{}")
        try:
            relation_info = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            relation_info = {}

        if not isinstance(relation_info, dict):
            continue

        relation_set = set()
        for judgment in relation_info.values():
            if not isinstance(judgment, dict):
                continue
            relation_type = _normalize_query_relation_type(
                judgment.get("query_relation_type", judgment.get("type", ""))
            )
            if relation_type in relation_label_map:
                all_relations.append(relation_type)
                relation_set.add(relation_type)

        if relation_set:
            user_query_relation_rows.append(
                {
                    "user_query": row.get("user_query", ""),
                    "conv_id": row.get("conv_id", ""),
                    "turn_id": row.get("turn_id", ""),
                    "relation_set": relation_set,
                }
            )

    def _relation_set_key(relation_set):
        return "+".join(
            relation for relation in relation_order if relation in relation_set
        )

    relation_set_counts = Counter(
        _relation_set_key(item["relation_set"])
        for item in user_query_relation_rows
    )
    venn_counts = {
        "extractive": relation_set_counts["extractive"],
        "abstractive": relation_set_counts["abstractive"],
        "discretionary": relation_set_counts["discretionary"],
        "extractive+abstractive": relation_set_counts["extractive+abstractive"],
        "extractive+discretionary": relation_set_counts["extractive+discretionary"],
        "abstractive+discretionary": relation_set_counts["abstractive+discretionary"],
        "extractive+abstractive+discretionary": relation_set_counts[
            "extractive+abstractive+discretionary"
        ],
    }
    total_user_queries = sum(venn_counts.values())
    assignment_df = pd.DataFrame(
        [
            {
                "user_query": item["user_query"],
                "conv_id": item["conv_id"],
                "turn_id": item["turn_id"],
                "relation_set": _relation_set_key(item["relation_set"]),
                "relation_set_display": " + ".join(
                    relation_label_map[relation]
                    for relation in relation_order
                    if relation in item["relation_set"]
                ),
            }
            for item in user_query_relation_rows
        ]
    )
    assignment_path = (
        f"{OUTPUT_PATH}/metadata/"
        f"{assignment_file_name}"
    )
    assignment_df.to_csv(
        assignment_path,
        index=False,
    )

    print("User queries assigned to Venn regions:", total_user_queries)
    print(pd.Series(venn_counts).rename_axis("region").reset_index(name="count"))

    relation_counts = (
        pd.Series(all_relations, dtype=str)
        .value_counts()
        .reindex(relation_order, fill_value=0)
        .rename_axis("query_relation_type")
        .reset_index(name="count")
    )
    relation_counts["query_relation_display"] = relation_counts[
        "query_relation_type"
    ].map(relation_label_map)

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=relation_counts["query_relation_display"],
            values=relation_counts["count"],
            textinfo="label+percent",
            textposition="inside",
            automargin=True,
            hovertemplate=(
                "%{label}<br>Issued web queries: %{value}"
                "<br>Share: %{percent}<extra></extra>"
            ),
            sort=False,
        )
    )
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 12), legend_pos=None)
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    venn_fig = go.Figure()
    circle_specs = [
        {
            "x0": 0.0,
            "y0": 0.9,
            "x1": 2.6,
            "y1": 3.5,
            "color": "rgba(99,110,250,0.35)",
        },
        {
            "x0": 1.6,
            "y0": 0.9,
            "x1": 4.2,
            "y1": 3.5,
            "color": "rgba(239,85,59,0.35)",
        },
        {
            "x0": 0.8,
            "y0": -0.2,
            "x1": 3.4,
            "y1": 2.4,
            "color": "rgba(0,204,150,0.35)",
        },
    ]
    for spec in circle_specs:
        venn_fig.add_shape(
            type="circle",
            x0=spec["x0"],
            y0=spec["y0"],
            x1=spec["x1"],
            y1=spec["y1"],
            line=dict(color="rgba(0,0,0,0.55)", width=2),
            fillcolor=spec["color"],
        )

    annotations = [
        (0.7, 2.45, str(venn_counts["extractive"])),
        (3.5, 2.45, str(venn_counts["abstractive"])),
        (2.1, 0.35, str(venn_counts["discretionary"])),
        (2.1, 2.55, str(venn_counts["extractive+abstractive"])),
        (1.35, 1.4, str(venn_counts["extractive+discretionary"])),
        (2.85, 1.4, str(venn_counts["abstractive+discretionary"])),
        (2.1, 1.85, str(venn_counts["extractive+abstractive+discretionary"])),
        (0.6, 3.7, "Extractive"),
        (3.6, 3.7, "Abstractive"),
        (2.1, -0.4, "Discretionary"),
    ]
    for x, y, text in annotations:
        venn_fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=18, color="black"),
        )

    venn_fig.update_layout(
        xaxis=dict(visible=False, range=[-0.2, 4.4]),
        yaxis=dict(visible=False, range=[-1.0, 4.2], scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    venn_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{venn_file_name}.html")
    venn_fig = with_paper_style(venn_fig, config=styler(18, 16), legend_pos=None)
    venn_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{venn_file_name}.pdf", format="pdf")


def plot_user_web_query_relation_distribution():
    df = pd.read_csv(
        f"{OUTPUT_PATH}/metadata/query_reformulations_user_web_query_relations.csv"
    )
    _plot_user_web_query_relation_distribution_from_df(df)



if __name__ == "__main__":
    # web_df = load_web_data_from_file(fmt="pkl")
    # print(f"Loaded web data: {len(web_df)}")
    # gather_query_reform_effective_factors(web_df)
    # print(preprocess_text("relational relations relationship.", stem=True))
    # print(preprocess_text("relational relations relationship.", stem=False))

    # compute_semantic_and_syntactic_similarity()
    # plot_semantic_and_syntactic_similarity()

    # web_query_tokens_source_detection()
    # plot_web_query_tokens_source_detection()

    # check_retrieved_source_effect_for_one_loop()
    # distribution_of_web_query_and_thoughts_over_time()

    # plot_number_of_loops_histogram()

    # reasons_for_another_web_query()
    plot_reasons_for_another_web_query_distribution()

    # type_of_user_and_web_query()
    plot_user_and_web_query_type_distribution()

    # count_models_with_web_queries()
    
    # plot_web_query_tokens_source_detection_over_time()
    # plot_query_term_count_trends_over_time(remove_stopwords=False)
    pass

# User Query:
# 140.42669472610166
# 1104.4129089595692
# User Query with history:
# 1730.9975129137172
# 7643.3799089092645
# Search Query:
# 55.10612532658732
# 25.863563513335
