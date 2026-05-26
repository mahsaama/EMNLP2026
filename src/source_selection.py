import os
import sys
import csv
import ast
import json
import re
from tqdm import tqdm
import pandas as pd
import socket
import ssl
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import ttest_rel
import numpy as np

pio.defaults.mathjax = None
from utils import *
from data_utils import *
from paper import with_paper_style, styler
from data_extraction import load_web_data_from_file, load_whole_data_from_file
from response_generation import _load_response_source_similarity_input


CONF = "emnlp/source_selection"
TIMEOUT = 5
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}


def extract_retrieved_safe_cited_source(web_df):
    outer_pattern = r"\ue200(?=[^\ue201]*\ue202[A-Za-z]+\d+[A-Za-z]+\d+(?:\ue202|\ue201))[^\ue201]*\ue201"
    inner_pattern = r"\ue202[A-Za-z]+(\d+)[A-Za-z]+(\d+)(?=\ue202|\ue201)"

    def _dedupe_cited_items(items):
        def _item_richness(item):
            score = 0
            for value in item.values():
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                score += 1
            return score

        unique_items = []
        key_to_index = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            dedupe_key = (
                item.get("url", ""),
                item.get("ref_index", None),
                item.get("turn_index", None),
            )
            existing_index = key_to_index.get(dedupe_key)
            if existing_index is None:
                key_to_index[dedupe_key] = len(unique_items)
                unique_items.append(item)
                continue

            if _item_richness(item) > _item_richness(unique_items[existing_index]):
                unique_items[existing_index] = item
        return unique_items

    web_df["srcs_retrieved"] = [{}] * len(web_df)
    web_df["srcs_safe_urls"] = [{}] * len(web_df)
    web_df["srcs_cited"] = [{}] * len(web_df)

    for i, row in tqdm(web_df.iterrows()):
        msgs = json.loads(row['turn_msgs'])
        srcs_retrieved = []
        srcs_safe_urls = []
        srcs_cited = []
        for msg in msgs:
            # retrieved
            retrieved = msg.get('metadata', {}).get('search_result_groups', [])
            for r in retrieved:
                entries = r.get("entries", [])
                for entry in entries:
                    url = entry.get("url", "")
                    if url:
                        d = urlparse(entry['url']).netloc.replace("www.", "")
                        srcs_retrieved.append(
                            {
                                "url": url,
                                "domain": d,
                                "title": entry.get("title", ""),
                                "ref_index": entry.get("ref_id", {}).get("ref_index", None) if entry.get("ref_id", {}) else None,
                                "turn_index": entry.get("ref_id", {}).get("turn_index", None) if entry.get("ref_id", {}) else None,
                                "snippet": entry["snippet"],
                            }
                        )

            retrieved = msg.get('metadata', {}).get('image_results', [])
            for ri, r in enumerate(retrieved):
                d = urlparse(r["url"]).netloc.replace("www.", "")
                srcs_retrieved.append(
                    {
                        "url": r["url"],
                        "domain": d,
                        "title": r.get("title", ""),
                        "ref_index": ri,
                    }
                )

            # safe urls
            safe_urls = msg.get('metadata', {}).get('safe_urls', [])
            for r in safe_urls:
                if r:
                    url = r.removesuffix("?utm_source=chatgpt.com").removesuffix("&utm_source=chatgpt.com")
                    d = urlparse(url).netloc.replace("www.", "")
                    srcs_safe_urls.append({
                        "url": url,
                        "domain": d
                    })

            # cited
            cited = msg.get('metadata', {}).get('content_references', [])
            for r in cited:
                matched_text = r.get("matched_text", "").strip()
                if matched_text:
                    outer = re.search(outer_pattern, matched_text)
                    if outer:
                        found_refs = re.findall(inner_pattern, outer.group(0))
                        cited_turns = []
                        cited_ranks = []
                        for fr in found_refs:
                            cited_turns.append(int(fr[0]))
                            cited_ranks.append(int(fr[1]))

                        # print(matched_text, cited_turns, cited_ranks)

                        url = r.get("url", "")
                        if url:
                            url = url.removesuffix("?utm_source=chatgpt.com").removesuffix("&utm_source=chatgpt.com")
                            d = urlparse(url).netloc.replace("www.", "")
                            srcs_cited.append(
                                {
                                    "url": url,
                                    "domain": d,
                                    "title": r.get("title"),
                                    "snippet": r.get("snippet"),
                                    "ref_index": cited_ranks[0],
                                    "turn_index": cited_turns[0]
                                }
                            )

                            
                        if "fallback_items" in r and r["fallback_items"]:
                            keys_to_check = ["images", "fallback_items"]
                        else:
                            keys_to_check = ["images", "items"]

                        for key in keys_to_check:
                            items = r.get(key, [])
                            refs = r.get("refs", [])
                            if items:
                                for ii, item in enumerate(items):
                                    url = item.get("url", "").removesuffix("?utm_source=chatgpt.com").removesuffix("&utm_source=chatgpt.com")
                                    d = urlparse(url).netloc.replace("www.", "")
                                    if item.get("refs", []):
                                        ref = item.get("refs", [])[0]
                                    else:
                                        ref = refs[ii] if ii < len(refs) else {}
                                    if url:
                                        srcs_cited.append(
                                            {
                                                "url": url,
                                                "domain": d,
                                                "title": item.get("title", ""),
                                                "snippet": item.get("snippet", ""),
                                                "ref_index": ref.get("ref_index", None),
                                                "turn_index": ref.get("turn_index", None)
                                            }
                                        )


        web_df.at[i, "srcs_retrieved"] = srcs_retrieved
        web_df.at[i, "srcs_safe_urls"] = srcs_safe_urls
        web_df.at[i, "srcs_cited"] = _dedupe_cited_items(srcs_cited)

    web_df.drop(columns=["turn_msgs"], inplace=True)
    web_df.reset_index(drop=True, inplace=True)
    
    web_df.to_csv(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.csv",
        index=False,
    )
    web_df.to_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    )

def _unique_source_count(items, key="url"):
    if not isinstance(items, list):
        return 0
    values = {
        item.get(key, "")
        for item in items
        if isinstance(item, dict) and item.get(key, "")
    }
    return len(values)


def _primary_model(models):
    if not isinstance(models, list):
        return "Unknown"
    cleaned = [model for model in models if isinstance(model, str) and model]
    if not cleaned:
        return "Unknown"
    return cleaned[-1]


def _prepare_source_count_df():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    df["model"] = df["openai_models"].apply(_primary_model)
    df["num_retrieved_urls"] = df["srcs_retrieved"].apply(_unique_source_count)
    df["num_safe_urls"] = df["srcs_safe_urls"].apply(_unique_source_count)
    df["num_cited_urls"] = df["srcs_cited"].apply(_unique_source_count)
    return df


def count_unique_retrieved_safe_cited():
    df = _prepare_source_count_df()
    return {
        "retrieved_urls": (df["num_retrieved_urls"] > 0).sum(),
        "safe_urls": (df["num_safe_urls"] > 0).sum(),
        "cited_urls": (df["num_cited_urls"] > 0).sum(),
    }


def compute_subset_counts(df, retrieved_col, safe_col, cited_col, key="url"):
    counts = {}

    for _, row in df.iterrows():
        retrieved = {
            item.get(key, "")
            for item in row[retrieved_col]
            if isinstance(item, dict) and item.get(key, "")
        }
        safe = {
            item.get(key, "")
            for item in row[safe_col]
            if isinstance(item, dict) and item.get(key, "")
        }
        cited = {
            item.get(key, "")
            for item in row[cited_col]
            if isinstance(item, dict) and item.get(key, "")
        }

        safe_in_retrieved = safe.issubset(retrieved)
        cited_in_safe = cited.issubset(safe)
        cited_in_retrieved = cited.issubset(retrieved)

        condition = (safe_in_retrieved, cited_in_safe, cited_in_retrieved)
        counts[condition] = counts.get(condition, 0) + 1

    return counts


def plot_subset_condition_counts():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()

    def _compute_cited_subset_counts(df, retrieved_col, cited_col, key="url"):
        counts = {False: 0, True: 0}
        for _, row in df.iterrows():
            retrieved = {
                item.get(key, "")
                for item in row[retrieved_col]
                if isinstance(item, dict) and item.get(key, "")
            }
            cited = {
                item.get(key, "")
                for item in row[cited_col]
                if isinstance(item, dict) and item.get(key, "")
            }
            counts[cited.issubset(retrieved)] += 1
        return counts

    url_counts = _compute_cited_subset_counts(
        df, "srcs_retrieved", "srcs_cited", key="url"
    )
    domain_counts = _compute_cited_subset_counts(
        df, "srcs_retrieved", "srcs_cited", key="domain"
    )

    label_map = {
        False: "C⊄R",
        True: "C⊆R",
    }

    rows = []
    for condition, label in [(False, label_map[False]), (True, label_map[True])]:
        rows.append(
            {"Condition": label, "Count": url_counts.get(condition, 0), "Type": "URLs"}
        )
        rows.append(
            {
                "Condition": label,
                "Count": domain_counts.get(condition, 0),
                "Type": "Domains",
            }
        )

    plot_df = pd.DataFrame(rows)
    order = (
        plot_df.groupby("Condition")["Count"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    plot_df["Condition"] = pd.Categorical(
        plot_df["Condition"], categories=order, ordered=True
    )
    plot_df = plot_df.sort_values("Condition")

    fig = go.Figure()
    for source_type in ["URLs", "Domains"]:
        subset = plot_df[plot_df["Type"] == source_type]
        fig.add_trace(
            go.Bar(
                x=subset["Condition"],
                y=subset["Count"],
                name=source_type,
                text=subset["Count"],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Subset Conditions",
        yaxis_title="#Turns",
    )
    file_name = "subset_relations_urls_and_domains"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18))
    fig.update_xaxes(tickfont=dict(size=14))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def save_topic_to_domains_json():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()

    topic_to_domains = {}
    for _, row in df.iterrows():
        topic = row.get("topic", "Other")
        if pd.isna(topic):
            topic = "Other"
        if str(topic).strip().lower() == "other":
            continue

        if topic not in topic_to_domains:
            topic_to_domains[topic] = set()

        topic_to_domains[topic].update(
            item.get("domain", "")
            for item in row["srcs_retrieved"]
            if isinstance(item, dict) and item.get("domain", "")
        )
        topic_to_domains[topic].update(
            item.get("domain", "")
            for item in row["srcs_safe_urls"]
            if isinstance(item, dict) and item.get("domain", "")
        )
        topic_to_domains[topic].update(
            item.get("domain", "")
            for item in row["srcs_cited"]
            if isinstance(item, dict) and item.get("domain", "")
        )

    topic_to_domains = {
        topic: sorted(values)
        for topic, values in topic_to_domains.items()
    }

    to_json(
        topic_to_domains,
        f"{OUTPUT_PATH}/metadata/topic_to_domains.json",
    )


def _normalize_domain_for_top_plots(domain):
    if not isinstance(domain, str):
        return ""
    normalized = domain.strip().lower().rstrip(".")
    if normalized.startswith("www."):
        normalized = normalized[4:]
    if normalized == "wikipedia.org" or normalized.endswith(".wikipedia.org"):
        return "wikipedia.org"
    return normalized


def _normalize_url_for_source_matching(url):
    if not isinstance(url, str):
        return ""
    normalized = (
        url.strip()
        .removesuffix("?utm_source=chatgpt.com")
        .removesuffix("&utm_source=chatgpt.com")
        .rstrip("/")
    )
    return normalized


def _domain_counter(df, col_name, top_k=20):

    counts = {}
    for items in df[col_name]:
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            domain = _normalize_domain_for_top_plots(item.get("domain", ""))
            if domain:
                counts[domain] = counts.get(domain, 0) + 1

    plot_df = pd.DataFrame(
        {"domain": list(counts.keys()), "count": list(counts.values())}
    )
    if len(plot_df) == 0:
        return plot_df
    plot_df = plot_df.sort_values("count", ascending=False)
    if top_k is None:
        return plot_df
    return plot_df.head(top_k)


def _url_count_for_source_column(df, col_name):
    if col_name not in df.columns:
        return 0

    total = 0
    for items in df[col_name]:
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            url = _normalize_url_for_source_matching(item.get("url", ""))
            if url:
                total += 1
    return int(total)


def _cited_domain_counter_split(df, top_k=20):
    external_counts = {}
    internal_counts = {}

    for _, row in df.iterrows():
        retrieved_items = row.get("srcs_retrieved", [])
        cited_items = row.get("srcs_cited", [])
        if not isinstance(retrieved_items, list) or not isinstance(cited_items, list):
            continue

        retrieved_urls = {
            _normalize_url_for_source_matching(item.get("url", ""))
            for item in retrieved_items
            if isinstance(item, dict) and item.get("url", "")
        }

        for item in cited_items:
            if not isinstance(item, dict):
                continue
            cited_url = _normalize_url_for_source_matching(item.get("url", ""))
            if not cited_url:
                # Cannot classify as external/internal without a cited URL.
                continue

            domain = _normalize_domain_for_top_plots(item.get("domain", ""))
            if not domain:
                domain = _normalize_domain_for_top_plots(urlparse(cited_url).netloc)
            if not domain:
                continue

            # External/internal split must be determined via URL overlap only.
            is_external = cited_url in retrieved_urls

            if is_external:
                external_counts[domain] = external_counts.get(domain, 0) + 1
            else:
                internal_counts[domain] = internal_counts.get(domain, 0) + 1

    domains = sorted(set(external_counts.keys()) | set(internal_counts.keys()))
    if not domains:
        return pd.DataFrame(
            columns=["domain", "external_count", "internal_count", "total_count"]
        )

    rows = []
    for domain in domains:
        external = int(external_counts.get(domain, 0))
        internal = int(internal_counts.get(domain, 0))
        rows.append(
            {
                "domain": domain,
                "external_count": external,
                "internal_count": internal,
                "total_count": external + internal,
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("total_count", ascending=False)
    if top_k is None:
        return plot_df
    return plot_df.head(top_k)


def _load_domain_plot_df():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()
    df = df[
        df["srcs_retrieved"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        & df["srcs_cited"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()
    return df


def _write_figure(
    fig,
    output_dir,
    file_name,
    paper_config=styler(18, 12),
    legend_pos=(0.8, 1.2),
    new_legend=None,
    x_tickfont_size=None,
    y_tickfont_size=10,
):
    os.makedirs(output_dir, exist_ok=True)
    # fig.write_html(f"{output_dir}/{file_name}.html")
    fig = with_paper_style(
        fig,
        config=paper_config,
        legend_pos=legend_pos,
        new_legend=new_legend,
    )
    if x_tickfont_size is None:
        if "20" in file_name:
            x_tickfont_size = 10
        else:
            x_tickfont_size = 8
    fig.update_xaxes(tickfont=dict(size=x_tickfont_size))
    fig.update_yaxes(tickfont=dict(size=y_tickfont_size))
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")


def plot_top_domains(separate_cited_external_internal=False):
    df = _load_domain_plot_df()

    if separate_cited_external_internal:
        subplot_titles = [
            "Top Retrieved Domains",
            "Top Cited Retrieved Domains",
            "Top Cited Parametric Domains",
        ]
    else:
        subplot_titles = [
            "Top Retrieved Domains",
            "Top Cited Domains",
        ]

    fig = make_subplots(
        rows=len(subplot_titles),
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.25,
    )

    retrieved_all_df = _domain_counter(df, "srcs_retrieved", top_k=None)
    retrieved_denominator = float(retrieved_all_df["count"].sum()) if len(retrieved_all_df) > 0 else 0.0
    retrieved_df = retrieved_all_df.head(20)
    if retrieved_denominator > 0 and len(retrieved_df) > 0:
        fig.add_trace(
            go.Bar(
                x=retrieved_df["domain"],
                y=retrieved_df["count"] / retrieved_denominator,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if separate_cited_external_internal:
        split_df = _cited_domain_counter_split(df, top_k=None)
        cited_external_denominator = (
            float(split_df["external_count"].sum()) if len(split_df) > 0 else 0.0
        )
        cited_internal_denominator = (
            float(split_df["internal_count"].sum()) if len(split_df) > 0 else 0.0
        )

        external_df = split_df[split_df["external_count"] > 0].copy()
        external_df = external_df.sort_values("external_count", ascending=False).head(20)
        if cited_external_denominator > 0 and len(external_df) > 0:
            fig.add_trace(
                go.Bar(
                    x=external_df["domain"],
                    y=external_df["external_count"] / cited_external_denominator,
                    marker_color="#00CC96",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        internal_df = split_df[split_df["internal_count"] > 0].copy()
        internal_df = internal_df.sort_values("internal_count", ascending=False).head(20)
        if cited_internal_denominator > 0 and len(internal_df) > 0:
            fig.add_trace(
                go.Bar(
                    x=internal_df["domain"],
                    y=internal_df["internal_count"] / cited_internal_denominator,
                    marker_color="#E45756",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
    else:
        cited_all_df = _domain_counter(df, "srcs_cited", top_k=None)
        cited_denominator = float(cited_all_df["count"].sum()) if len(cited_all_df) > 0 else 0.0
        cited_df = cited_all_df.head(20)
        if cited_denominator > 0 and len(cited_df) > 0:
            fig.add_trace(
                go.Bar(
                    x=cited_df["domain"],
                    y=cited_df["count"] / cited_denominator,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        # height=1000,
        margin=dict(l=70, b=60, t=30, r=40),
    )
    fig.update_xaxes(
        tickangle=-30,
        automargin=True,
    )
    fig.add_annotation(
        x=-0.09,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Percentage of URLs",
        textangle=-90,
        showarrow=False,
        font=dict(size=18, color="black"),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = (
        "top_domains_overall_split_cited"
        if separate_cited_external_internal
        else "top_domains_overall"
    )
    _write_figure(
        fig,
        f"{OUTPUT_PATH}/{CONF}",
        file_name,
        styler(18, 18),
        legend_pos=None,
    )


def _plot_top_domains_for_subset(
    df,
    subset_label,
    output_dir,
    separate_cited_external_internal=False,
    use_plot_top_domains_setup=False,
):
    if separate_cited_external_internal:
        subplot_titles = [
            "Top Retrieved Domains",
            "Top Cited Retrieved Domains",
            "Top Cited Parametric Domains",
        ]
    else:
        subplot_titles = [
            "Top Retrieved Domains",
            "Top Cited Domains",
        ]

    fig = make_subplots(
        rows=len(subplot_titles),
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.3,
    )

    has_data = False
    retrieved_all_df = _domain_counter(df, "srcs_retrieved", top_k=None)
    retrieved_denominator = float(retrieved_all_df["count"].sum()) if len(retrieved_all_df) > 0 else 0.0
    retrieved_df = retrieved_all_df.head(20)
    if retrieved_denominator > 0 and len(retrieved_df) > 0:
        has_data = True
        fig.add_trace(
            go.Bar(
                x=retrieved_df["domain"],
                y=retrieved_df["count"] / retrieved_denominator,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if separate_cited_external_internal:
        split_df = _cited_domain_counter_split(df, top_k=None)
        cited_external_denominator = (
            float(split_df["external_count"].sum()) if len(split_df) > 0 else 0.0
        )
        cited_internal_denominator = (
            float(split_df["internal_count"].sum()) if len(split_df) > 0 else 0.0
        )

        external_df = split_df[split_df["external_count"] > 0].copy()
        external_df = external_df.sort_values("external_count", ascending=False).head(20)
        if cited_external_denominator > 0 and len(external_df) > 0:
            has_data = True
            fig.add_trace(
                go.Bar(
                    x=external_df["domain"],
                    y=external_df["external_count"] / cited_external_denominator,
                    marker_color="#00CC96",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        internal_df = split_df[split_df["internal_count"] > 0].copy()
        internal_df = internal_df.sort_values("internal_count", ascending=False).head(20)
        if cited_internal_denominator > 0 and len(internal_df) > 0:
            has_data = True
            fig.add_trace(
                go.Bar(
                    x=internal_df["domain"],
                    y=internal_df["internal_count"] / cited_internal_denominator,
                    marker_color="#E45756",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
    else:
        cited_all_df = _domain_counter(df, "srcs_cited", top_k=None)
        cited_denominator = float(cited_all_df["count"].sum()) if len(cited_all_df) > 0 else 0.0
        cited_df = cited_all_df.head(20)
        if cited_denominator > 0 and len(cited_df) > 0:
            has_data = True
            fig.add_trace(
                go.Bar(
                    x=cited_df["domain"],
                    y=cited_df["count"] / cited_denominator,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
    if not has_data:
        return

    fig.update_layout(
        # height=1000,
        margin=dict(l=70, b=60, t=30, r=40),
    )
    fig.update_xaxes(
        tickangle=-30,
        automargin=True,
    )
    fig.add_annotation(
        x=-0.12,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Percentage of URLs",
        textangle=-90,
        showarrow=False,
        font=dict(size=18, color="black"),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = (
        "top_20_domains_split_cited"
        if separate_cited_external_internal
        else "top_20_domains"
    )
    x_tickfont_size = None
    if use_plot_top_domains_setup:
        x_tickfont_size = 8
    _write_figure(
        fig,
        output_dir,
        file_name,
        styler(18, 18),
        legend_pos=None,
        x_tickfont_size=x_tickfont_size,
    )


def plot_top_domains_by_selected_topics(separate_cited_external_internal=False):
    df = _load_domain_plot_df()
    # selected_topics = ["Health", "Travel", "Finance", "Politics & History", "Science"]
    selected_topics = set(df["topic"].unique())

    for topic in selected_topics:
        topic_df = df[df["topic"] == topic].copy()
        if len(topic_df) == 0:
            continue
        safe_topic = re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_").lower()
        output_dir = f"{OUTPUT_PATH}/{CONF}/top_domains_by_topic/{safe_topic}"
        _plot_top_domains_for_subset(
            topic_df,
            topic,
            output_dir,
            separate_cited_external_internal=separate_cited_external_internal,
            use_plot_top_domains_setup=True,
        )


def plot_top_domains_by_model(separate_cited_external_internal=False):
    df = _load_domain_plot_df()
    df["model"] = df["openai_models"].apply(_primary_model)
    df = df[df["model"].str.lower() != "unknown"].copy()

    for model in sorted(df["model"].dropna().unique()):
        model_df = df[df["model"] == model].copy()
        if len(model_df) == 0:
            continue
        safe_model = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
        output_dir = f"{OUTPUT_PATH}/{CONF}/top_domains_by_model/{safe_model}"
        _plot_top_domains_for_subset(
            model_df,
            model,
            output_dir,
            separate_cited_external_internal=separate_cited_external_internal,
        )


def compare_domain_reliability_by_source_type(model_name="gpt-4o-mini", temperature=0.0):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()
    results_path = (
        f"{OUTPUT_PATH}/{CONF}/domain_reliability_evaluation/"
        f"{model_name}/{temperature}/results.json"
    )
    reliability_results = load_json(results_path)

    reliability_lookup = {}
    for topic, topic_results in reliability_results.items():
        reliability_lookup[topic] = {}
        for domain, raw_result in topic_results.items():
            score = None
            if isinstance(raw_result, str) and raw_result.strip():
                try:
                    score = json.loads(raw_result).get("reliability_score")
                except json.JSONDecodeError:
                    score = None
            elif isinstance(raw_result, dict):
                if "reliability_score" in raw_result:
                    score = raw_result.get("reliability_score")
                elif "Output" in raw_result and raw_result["Output"]:
                    try:
                        score = json.loads(raw_result["Output"]).get("reliability_score")
                    except json.JSONDecodeError:
                        score = None
            if score is not None:
                reliability_lookup[topic][domain] = float(score)

    def _avg_score(row, col_name):
        topic_scores = reliability_lookup.get(row["topic"], {})
        domains = sorted(
            {
                item.get("domain", "")
                for item in row[col_name]
                if isinstance(item, dict) and item.get("domain", "")
            }
        )
        scores = [topic_scores[d] for d in domains if d in topic_scores]
        if not scores:
            return np.nan
        return float(np.mean(scores))

    df["retrieved_reliability"] = df.apply(
        lambda row: _avg_score(row, "srcs_retrieved"), axis=1
    )
    df["cited_reliability"] = df.apply(
        lambda row: _avg_score(row, "srcs_cited"), axis=1
    )

    plot_df = df.dropna(subset=["retrieved_reliability", "cited_reliability"]).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=plot_df["retrieved_reliability"],
            name="Retrieved",
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Violin(
            y=plot_df["cited_reliability"],
            name="Cited",
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Source Type",
        yaxis_title="Average Domain Reliability",
    )
    fig.update_yaxes(range=[1, 5])

    file_name = "domain_reliability_by_source_type"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 14))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    ttest_results = {}
    comparisons = [
        ("retrieved_reliability", "cited_reliability", "retrieved_vs_cited"),
    ]
    for left_col, right_col, label in comparisons:
        paired = plot_df[[left_col, right_col]].dropna()
        if len(paired) == 0:
            continue
        stat, pvalue = ttest_rel(paired[left_col], paired[right_col])
        ttest_results[label] = {
            "n": int(len(paired)),
            "mean_left": float(paired[left_col].mean()),
            "mean_right": float(paired[right_col].mean()),
            "t_statistic": float(stat),
            "p_value": float(pvalue),
        }

    to_json(
        ttest_results,
        f"{OUTPUT_PATH}/{CONF}/{file_name}_paired_ttests.json",
    )
    plot_df[
        [
            "user_id",
            "conv_id",
            "turn_id",
            "topic",
            "retrieved_reliability",
            "cited_reliability",
        ]
    ].to_csv(
        f"{OUTPUT_PATH}/{CONF}/{file_name}_samples.csv",
        index=False,
    )

    return ttest_results


def plot_url_counts_over_time(separate_cited_external_internal=False):
    df = _prepare_source_count_df()

    if separate_cited_external_internal:
        def _count_cited_external_internal_urls(row):
            retrieved_urls = {
                _normalize_url_for_source_matching(item.get("url", ""))
                for item in row.get("srcs_retrieved", [])
                if isinstance(item, dict) and item.get("url", "")
            }
            cited_external_urls = set()
            cited_internal_urls = set()
            for item in row.get("srcs_cited", []):
                if not isinstance(item, dict):
                    continue
                cited_url = _normalize_url_for_source_matching(item.get("url", ""))
                if not cited_url:
                    continue
                if cited_url in retrieved_urls:
                    cited_external_urls.add(cited_url)
                else:
                    cited_internal_urls.add(cited_url)
            return pd.Series(
                {
                    "num_cited_external_urls": len(cited_external_urls),
                    "num_cited_internal_urls": len(cited_internal_urls),
                }
            )

        df = df.copy()
        df[["num_cited_external_urls", "num_cited_internal_urls"]] = df.apply(
            _count_cited_external_internal_urls,
            axis=1,
        )
        value_cols = [
            "num_retrieved_urls",
            "num_cited_external_urls",
            "num_cited_internal_urls",
        ]
    else:
        value_cols = ["num_retrieved_urls", "num_cited_urls"]

    monthly = (
        df.groupby("month")[value_cols]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly[("num_retrieved_urls", "mean")],
            mode="lines+markers",
            name="Retrieved URLs",
            error_y=dict(
                type="data",
                array=monthly[("num_retrieved_urls", "sem")].fillna(0),
                visible=True,
            ),
        )
    )
    if separate_cited_external_internal:
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly[("num_cited_internal_urls", "mean")],
                mode="lines+markers",
                name="Cited Unexplained/Internal URLs",
                error_y=dict(
                    type="data",
                    array=monthly[("num_cited_internal_urls", "sem")].fillna(0),
                    visible=True,
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly[("num_cited_external_urls", "mean")],
                mode="lines+markers",
                name="Cited Retrieved/External URLs",
                error_y=dict(
                    type="data",
                    array=monthly[("num_cited_external_urls", "sem")].fillna(0),
                    visible=True,
                ),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly[("num_cited_urls", "mean")],
                mode="lines+markers",
                name="Cited URLs",
                error_y=dict(
                    type="data",
                    array=monthly[("num_cited_urls", "sem")].fillna(0),
                    visible=True,
                ),
            )
        )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average # URLs per Turn",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    file_name = "url_counts_over_time"
    if separate_cited_external_internal:
        file_name += "_split_cited"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 20))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def plot_grounding_rate_violin_over_time():
    df = _prepare_source_count_df()

    rows = []
    for _, row in df.iterrows():
        retrieved_urls = {
            _normalize_url_for_source_matching(item.get("url", ""))
            for item in row.get("srcs_retrieved", [])
            if isinstance(item, dict) and item.get("url", "")
        }
        cited_urls = {
            _normalize_url_for_source_matching(item.get("url", ""))
            for item in row.get("srcs_cited", [])
            if isinstance(item, dict) and item.get("url", "")
        }
        cited_urls = {url for url in cited_urls if url}
        if len(cited_urls) == 0:
            continue

        cited_external_urls = cited_urls & retrieved_urls
        cited_internal_urls = cited_urls - retrieved_urls
        grounding_rate = len(cited_external_urls) / len(cited_urls)
        unexplained_rate = len(cited_internal_urls) / len(cited_urls)

        rows.append(
            {
                "user_id": row.get("user_id"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "time": row.get("time"),
                "month": row.get("month"),
                "num_cited_urls": int(len(cited_urls)),
                "num_cited_external_urls": int(len(cited_external_urls)),
                "num_cited_internal_urls": int(len(cited_internal_urls)),
                "grounding_rate": float(grounding_rate),
                "unexplained_rate": float(unexplained_rate),
            }
        )

    plot_df = pd.DataFrame(rows)
    if len(plot_df) == 0:
        return plot_df

    plot_df["month"] = pd.to_datetime(plot_df["month"], errors="coerce")
    plot_df = plot_df.dropna(subset=["month"]).sort_values("month")
    if len(plot_df) == 0:
        return plot_df

    plot_df["month_label"] = plot_df["month"].dt.strftime("%b %Y")
    month_order = (
        plot_df[["month", "month_label"]]
        .drop_duplicates()
        .sort_values("month")["month_label"]
        .tolist()
    )
    tickvals_every_2_months = month_order[::2] if month_order else []
    shown_months = set(tickvals_every_2_months)
    plot_df = plot_df[plot_df["month_label"].isin(shown_months)].copy()
    if len(plot_df) == 0:
        return plot_df
    shown_month_order = [m for m in month_order if m in shown_months]
    month_to_pos = {m: float(i) for i, m in enumerate(shown_month_order)}
    plot_df["month_pos"] = plot_df["month_label"].map(month_to_pos)
    x_offset = 0.15

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=plot_df["month_pos"] - x_offset,
            y=plot_df["grounding_rate"],
            name="Grounding Rate (Cited Retrieved/External)",
            marker_color="#00CC96",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            width=0.42,
        )
    )
    fig.add_trace(
        go.Violin(
            x=plot_df["month_pos"] + x_offset,
            y=plot_df["unexplained_rate"],
            name="Unexplained Rate (Cited Internal)",
            marker_color="#E45756",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            width=0.42,
        )
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Grounding Rate",
        xaxis=dict(
            tickmode="array",
            tickvals=[month_to_pos[m] for m in shown_month_order],
            ticktext=shown_month_order,
            tickangle=-30,
            range=[-0.6, max(len(shown_month_order) - 0.4, 0.6)],
        ),
        violinmode="overlay",
        margin=dict(b=120),
    )
    fig.update_yaxes(tickformat=".0%")

    file_name = "grounding_rate_violin_over_time"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.update_xaxes(
        tickangle=-30,
        tickmode="array",
        tickvals=[month_to_pos[m] for m in shown_month_order],
        ticktext=shown_month_order,
    )
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    monthly_stats = (
        plot_df.groupby(["month", "month_label"])
        .agg(
            grounding_rate_mean=("grounding_rate", "mean"),
            grounding_rate_std=("grounding_rate", "std"),
            unexplained_rate_mean=("unexplained_rate", "mean"),
            unexplained_rate_std=("unexplained_rate", "std"),
            num_turns=("grounding_rate", "count"),
        )
        .reset_index()
        .sort_values("month")
    )
    grounding_std = monthly_stats["grounding_rate_std"].fillna(0)
    unexplained_std = monthly_stats["unexplained_rate_std"].fillna(0)
    grounding_lower = (
        monthly_stats["grounding_rate_mean"] - grounding_std
    ).clip(lower=0, upper=1)
    grounding_upper = (
        monthly_stats["grounding_rate_mean"] + grounding_std
    ).clip(lower=0, upper=1)
    unexplained_lower = (
        monthly_stats["unexplained_rate_mean"] - unexplained_std
    ).clip(lower=0, upper=1)
    unexplained_upper = (
        monthly_stats["unexplained_rate_mean"] + unexplained_std
    ).clip(lower=0, upper=1)

    mean_std_fig = go.Figure()
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=grounding_lower,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=grounding_upper,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 204, 150, 0.2)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=monthly_stats["grounding_rate_mean"],
            mode="lines+markers",
            name="Grounding Rate (Cited Retrieved/External)",
            marker_color="#00CC96",
            line=dict(color="#00CC96"),
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=unexplained_lower,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=unexplained_upper,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(228, 87, 86, 0.2)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=monthly_stats["unexplained_rate_mean"],
            mode="lines+markers",
            name="Unexplained Rate (Cited Internal)",
            marker_color="#E45756",
            line=dict(color="#E45756"),
        )
    )
    mean_std_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Rate",
        xaxis=dict(
            categoryorder="array",
            categoryarray=shown_month_order,
            tickmode="array",
            tickvals=shown_month_order,
            tickangle=-30,
        ),
        margin=dict(b=120),
    )
    mean_std_fig.update_yaxes(tickformat=".0%")

    mean_std_file_name = "grounding_rate_mean_std_over_time"
    mean_std_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{mean_std_file_name}.html")
    mean_std_fig = with_paper_style(mean_std_fig, config=styler(18, 16))
    mean_std_fig.update_xaxes(
        tickangle=-30,
        tickmode="array",
        tickvals=shown_month_order,
    )
    mean_std_fig.write_image(
        f"{OUTPUT_PATH}/{CONF}/{mean_std_file_name}.pdf",
        format="pdf",
    )
    monthly_stats.to_csv(
        f"{OUTPUT_PATH}/{CONF}/{mean_std_file_name}_monthly_stats.csv",
        index=False,
    )

    plot_df[
        [
            "user_id",
            "conv_id",
            "turn_id",
            "time",
            "month",
            "num_cited_urls",
            "num_cited_external_urls",
            "num_cited_internal_urls",
            "grounding_rate",
            "unexplained_rate",
        ]
    ].to_csv(
        f"{OUTPUT_PATH}/{CONF}/{file_name}_samples.csv",
        index=False,
    )

    return plot_df


def plot_retrieved_url_counts_over_time_by_model():
    df = _prepare_source_count_df()
    df = df[df["model"].str.lower() != "unknown"].copy()
    df = df.dropna(subset=["month"])
    if len(df) == 0:
        return

    monthly = (
        df.groupby(["month", "model"])["num_retrieved_urls"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .sort_values(["model", "month"])
    )
    if len(monthly) == 0:
        return

    model_order = (
        monthly.groupby("model")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig = go.Figure()
    for model in model_order:
        model_df = monthly[monthly["model"] == model]
        if len(model_df) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=model_df["month"],
                y=model_df["mean"],
                mode="lines+markers",
                name=model,
                error_y=dict(
                    type="data",
                    array=model_df["sem"].fillna(0),
                    visible=True,
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average # Retrieved URLs per Turn",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    file_name = "retrieved_url_counts_over_time_by_model"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18), legend_pos=(0.8, 1.8))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def _plot_url_counts_grouped(df, group_col, file_name, xaxis_title):
    grouped = (
        df.groupby(group_col)[["num_retrieved_urls", "num_cited_urls"]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    grouped = grouped.sort_values(("num_cited_urls", "mean"), ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped[group_col],
            y=grouped[("num_retrieved_urls", "mean")],
            name="Retrieved URLs",
            error_y=dict(
                type="data",
                array=grouped[("num_retrieved_urls", "sem")].fillna(0),
                visible=True,
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped[group_col],
            y=grouped[("num_cited_urls", "mean")],
            name="Cited URLs",
            error_y=dict(
                type="data",
                array=grouped[("num_cited_urls", "sem")].fillna(0),
                visible=True,
            ),
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis_title=xaxis_title,
        yaxis_title="Average # URLs per Turn",
        xaxis=dict(tickangle=-30),
    )
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 20))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def plot_url_counts_by_model():
    df = _prepare_source_count_df()
    df = df[df["model"].str.lower() != "unknown"].copy()
    _plot_url_counts_grouped(
        df,
        group_col="model",
        file_name="url_counts_by_model",
        xaxis_title="Model",
    )


def plot_url_counts_by_topic():
    df = _prepare_source_count_df()
    df = df[df["topic"].fillna("").str.lower() != "other"].copy()
    _plot_url_counts_grouped(
        df,
        group_col="topic",
        file_name="url_counts_by_topic",
        xaxis_title="Topic",
    )


def compare_safe_vs_retrieved_minus_safe_reachability():
    cache_path = f"{OUTPUT_PATH}/metadata/safe_vs_retrieved_minus_safe_reachability.csv"
    if os.path.exists(cache_path):
        comparison_df = pd.read_csv(cache_path)
    else:
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
        ).copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            retrieved = {
                item.get("url", "")
                for item in row["srcs_retrieved"]
                if isinstance(item, dict) and item.get("url", "")
            }
            safe = {
                item.get("url", "")
                for item in row["srcs_safe_urls"]
                if isinstance(item, dict) and item.get("url", "")
            }

            safe_reachable = count_reachable_urls(safe)
            retrieved_minus_safe = sorted(retrieved - safe)
            retrieved_minus_safe_reachable = count_reachable_urls(retrieved_minus_safe)

            rows.append(
                {
                    "month": row["month"],
                    "safe_total": len(safe),
                    "safe_reachable": safe_reachable,
                    "retrieved_minus_safe_total": len(retrieved_minus_safe),
                    "retrieved_minus_safe_reachable": retrieved_minus_safe_reachable,
                }
            )

        comparison_df = pd.DataFrame(rows)
        comparison_df.to_csv(cache_path, index=False)

    comparison_df["month"] = pd.to_datetime(comparison_df["month"], errors="coerce")
    monthly = (
        comparison_df.groupby("month")[
            [
                "safe_total",
                "safe_reachable",
                "retrieved_minus_safe_total",
                "retrieved_minus_safe_reachable",
            ]
        ]
        .sum()
        .sort_index()
    )
    if len(monthly) == 0:
        return comparison_df

    full_month_range = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS",
    )
    monthly = monthly.reindex(full_month_range, fill_value=0).rename_axis("month").reset_index()
    monthly["safe_reachability_rate"] = (
        monthly["safe_reachable"] / monthly["safe_total"].replace(0, pd.NA)
    ).fillna(0)
    monthly["retrieved_minus_safe_reachability_rate"] = (
        monthly["retrieved_minus_safe_reachable"]
        / monthly["retrieved_minus_safe_total"].replace(0, pd.NA)
    ).fillna(0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["safe_reachability_rate"],
            mode="lines+markers",
            name="Safe URLs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["retrieved_minus_safe_reachability_rate"],
            mode="lines+markers",
            name="Retrieved - Safe URLs",
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Reachability Rate",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    file_name = "safe_vs_retrieved_minus_safe_reachability"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    return comparison_df


def plot_retrieved_safe_cited_positions(separate_cited_external_internal=False):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()

    def _valid_ref_index(value):
        try:
            rank = float(value)
        except (TypeError, ValueError):
            return np.nan
        if not np.isfinite(rank) or rank < 0:
            return np.nan
        # Source ranks are 0-indexed in metadata; shift by +1 for plotting so
        # log10(rank) starts at 0 instead of negative values.
        return rank + 1.0

    def _append_average_rank(rows, group_name, ranks):
        valid_ranks = [rank for rank in ranks if np.isfinite(rank)]
        if valid_ranks:
            rows.append(
                {
                    "group": group_name,
                    "average_rank": float(np.mean(valid_ranks)),
                }
            )

    plot_rows = []
    avg_rank_rows = []
    for _, row in df.iterrows():
        retrieved_urls = {
            _normalize_url_for_source_matching(item.get("url", ""))
            for item in row["srcs_retrieved"]
            if isinstance(item, dict) and item.get("url", "")
        }
        retrieved_ref_indices = []
        for item in row["srcs_retrieved"]:
            if not isinstance(item, dict):
                continue
            ref_index = _valid_ref_index(item.get("ref_index"))
            if not np.isfinite(ref_index):
                continue
            retrieved_ref_indices.append(ref_index)
            turn_index = item.get("turn_index")
            if turn_index is None:
                continue
            plot_rows.append(
                {
                    "group": "Retrieved URLs",
                    "ref_index": ref_index,
                    "turn_index": turn_index,
                }
            )

        cited_ref_indices = []
        cited_external_ref_indices = []
        cited_internal_ref_indices = []
        for item in row["srcs_cited"]:
            if not isinstance(item, dict):
                continue
            ref_index = _valid_ref_index(item.get("ref_index"))
            if not np.isfinite(ref_index):
                continue
            if separate_cited_external_internal:
                cited_url = _normalize_url_for_source_matching(item.get("url", ""))
                is_external = bool(cited_url and cited_url in retrieved_urls)
                if is_external:
                    group_name = "Cited Retrieved/External URLs"
                    cited_external_ref_indices.append(ref_index)
                else:
                    group_name = "Cited Unexplained/Internal URLs"
                    cited_internal_ref_indices.append(ref_index)
            else:
                group_name = "Cited URLs"
                cited_ref_indices.append(ref_index)
            turn_index = item.get("turn_index")
            if turn_index is None:
                continue
            plot_rows.append(
                {
                    "group": group_name,
                    "ref_index": ref_index,
                    "turn_index": turn_index,
                }
            )

        _append_average_rank(avg_rank_rows, "Retrieved URLs", retrieved_ref_indices)
        if separate_cited_external_internal:
            _append_average_rank(
                avg_rank_rows,
                "Cited Retrieved/External URLs",
                cited_external_ref_indices,
            )
            _append_average_rank(
                avg_rank_rows,
                "Cited Unexplained/Internal URLs",
                cited_internal_ref_indices,
            )
        else:
            _append_average_rank(avg_rank_rows, "Cited URLs", cited_ref_indices)

    plot_df = pd.DataFrame(plot_rows)
    avg_rank_df = pd.DataFrame(avg_rank_rows)
    if len(plot_df) == 0 and len(avg_rank_df) == 0:
        return

    if separate_cited_external_internal:
        rank_specs = [
            ("Retrieved URLs", "Retrieved", "#636EFA"),
            (
                "Cited Unexplained/Internal URLs",
                "Cited<br>Unexplained/Internal",
                "#E45756",
            ),
            (
                "Cited Retrieved/External URLs",
                "Cited<br>Retrieved/External",
                "#00CC96",
            ),
        ]
    else:
        rank_specs = [
            ("Retrieved URLs", "Retrieved", "#636EFA"),
            ("Cited URLs", "Cited", "#EF553B"),
        ]
    group_order = [group_name for group_name, _label, _color in rank_specs]

    if len(plot_df) > 0:
        aggregated_df = (
            plot_df.groupby(["group", "turn_index", "ref_index"])
            .size()
            .reset_index(name="count")
        )

        if len(group_order) == 1:
            offsets = [0.0]
        elif len(group_order) == 2:
            offsets = [-0.15, 0.15]
        elif len(group_order) == 3:
            offsets = [-0.2, 0.0, 0.2]
        else:
            offsets = np.linspace(-0.25, 0.25, len(group_order))
        x_offsets = {group_name: float(offset) for group_name, offset in zip(group_order, offsets)}

        fig = go.Figure()
        for group_name in group_order:
            subset = aggregated_df[aggregated_df["group"] == group_name]
            fig.add_trace(
                go.Scatter(
                    x=subset["turn_index"] + 1 + x_offsets[group_name],
                    y=subset["ref_index"],
                    mode="markers",
                    name=group_name,
                    marker=dict(
                        size=3,
                    ),
                    hovertemplate="Loop=%{x}<br>Rank=%{y}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Loop",
            yaxis_title="Rank",
        )
        fig.update_xaxes(dtick=5)
        fig.update_yaxes(autorange="reversed")
        file_name = "retrieved_safe_cited_positions"
        if separate_cited_external_internal:
            file_name += "_split_cited"
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 16))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    if len(avg_rank_df) == 0:
        return

    violin_fig = go.Figure()
    use_log10_violin = separate_cited_external_internal
    if use_log10_violin:
        valid_values = pd.to_numeric(
            avg_rank_df["average_rank"], errors="coerce"
        ).to_numpy()
        valid_values = valid_values[np.isfinite(valid_values)]
        valid_values = valid_values[valid_values > 0]
        if len(valid_values) == 0:
            return
        valid_log_values = np.log10(valid_values)
        main_rank_min = float(np.floor(np.min(valid_log_values)))
        main_rank_max = float(np.ceil(np.max(valid_log_values)))
        if not np.isfinite(main_rank_min):
            main_rank_min = 0.0
        if not np.isfinite(main_rank_max):
            main_rank_max = 1.0
        if main_rank_max <= main_rank_min:
            main_rank_max = main_rank_min + 1.0
    else:
        main_rank_min = 0.0
        main_rank_max = float(avg_rank_df["average_rank"].max())
        if not np.isfinite(main_rank_max) or main_rank_max <= 0:
            main_rank_max = 1.0
    for group_name, label, color in rank_specs:
        subset = avg_rank_df.loc[
            avg_rank_df["group"] == group_name, "average_rank"
        ].dropna()
        if len(subset) == 0:
            continue

        zoom_percentile = 85 if ("Cited" in label and "Internal" in label) or label == "Cited" else 90
        if use_log10_violin:
            subset = subset[subset > 0]
            if len(subset) == 0:
                continue
            subset_plot = np.log10(subset.to_numpy())
        else:
            subset_plot = subset.to_numpy()

        violin_fig.add_trace(
            go.Violin(
                x=[label] * len(subset_plot),
                y=subset_plot,
                name=label,
                marker_color=color,
                line_color=color,
                box_visible=True,
                width=0.9,
                meanline_visible=True,
                showlegend=True,
                legendgroup=label,
            )
        )
        # mean_value = float(np.mean(subset_plot))
        # median_value = float(np.median(subset_plot))
        # annotation_text = f"mean={mean_value:,.1f}<br>median={median_value:,.1f}"
        # annotation_y = float(np.nanpercentile(subset_plot, zoom_percentile))
        # if not np.isfinite(annotation_y):
        #     annotation_y = mean_value
        # violin_fig.add_annotation(
        #     x=label,
        #     y=min(main_rank_max - 0.05, annotation_y),
        #     text=annotation_text,
        #     showarrow=True,
        #     arrowhead=1,
        #     ax=35,
        #     ay=-25,
        #     font=dict(size=12, color=color),
        #     bgcolor="rgba(255,255,255,0.85)",
        #     bordercolor=color,
        # )

    violin_fig.update_layout(
        xaxis_title="Source Type",
        yaxis_title="Average Rank (log10)" if use_log10_violin else "Average Rank",
        xaxis=dict(
            categoryorder="array",
            categoryarray=[label for _group_name, label, _color in rank_specs],
            tickangle=0,
        ),
        yaxis=dict(range=[main_rank_min, main_rank_max], autorange=False),
        # height=680,
        # width=900,
    )

    file_name = "retrieved_safe_cited_positions_rank_violinplot"
    if separate_cited_external_internal:
        file_name += "_split_cited"
    violin_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    violin_fig = with_paper_style(violin_fig, config=styler(18, 16), legend_pos=None)
    violin_fig.update_xaxes(tickangle=0, tickfont=dict(size=16))
    violin_fig.update_layout(
        # height=680,
        # width=900,
        yaxis=dict(range=[main_rank_min, main_rank_max], autorange=False),
    )
    violin_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return []
        return parsed if isinstance(parsed, list) else []
    return []


def _as_valid_number(value, min_value=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    if min_value is not None and number < min_value:
        return np.nan
    return number


def _load_first_existing_pickle(paths):
    for path in paths:
        if os.path.exists(path):
            return pd.read_pickle(path).copy()
    raise FileNotFoundError(f"None of these files exist: {paths}")


def _build_retrieved_safe_cited_rank_plot_rows(
    df,
    retrieved_rank_col,
    cited_rank_col,
    rank_transform=None,
    min_rank=None,
):
    rank_transform = rank_transform or (lambda rank: rank)
    plot_rows = []

    for _, row in df.iterrows():
        retrieved_sources = _as_list(row.get("srcs_retrieved", []))
        retrieved_ranks = _as_list(row.get(retrieved_rank_col, []))
        cited_sources = _as_list(row.get("srcs_cited", []))
        cited_ranks = _as_list(row.get(cited_rank_col, []))

        for idx, item in enumerate(retrieved_sources):
            if not isinstance(item, dict):
                continue
            turn_index = item.get("turn_index")
            if turn_index is None:
                continue

            rank = np.nan
            if idx < len(retrieved_ranks):
                rank = rank_transform(_as_valid_number(retrieved_ranks[idx], min_rank))
            if np.isfinite(rank):
                plot_rows.append(
                    {
                        "group": "Retrieved URLs",
                        "rank": rank,
                        "turn_index": turn_index,
                    }
                )

        for idx, item in enumerate(cited_sources):
            if not isinstance(item, dict):
                continue
            turn_index = item.get("turn_index")
            if turn_index is None:
                continue
            rank = np.nan
            if idx < len(cited_ranks):
                rank = rank_transform(_as_valid_number(cited_ranks[idx], min_rank))
            if not np.isfinite(rank):
                continue
            plot_rows.append(
                {
                    "group": "Cited URLs",
                    "rank": rank,
                    "turn_index": turn_index,
                }
            )

    return plot_rows


def _plot_retrieved_safe_cited_ranks(plot_rows, file_name, yaxis_title):
    plot_df = pd.DataFrame(plot_rows)
    if len(plot_df) == 0:
        return

    aggregated_df = (
        plot_df.groupby(["group", "turn_index", "rank"])
        .size()
        .reset_index(name="count")
    )

    x_offsets = {
        "Retrieved URLs": -0.1,
        "Cited URLs": 0.1,
    }

    fig = go.Figure()
    for group_name in ["Retrieved URLs", "Cited URLs"]:
        subset = aggregated_df[aggregated_df["group"] == group_name]
        fig.add_trace(
            go.Scatter(
                x=subset["turn_index"] + 1 + x_offsets[group_name],
                y=subset["rank"],
                mode="markers",
                name=group_name,
                marker=dict(
                    size=3,
                ),
                hovertemplate="Loop=%{x}<br>Rank=%{y}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Loop",
        yaxis_title=yaxis_title,
    )
    fig.update_xaxes(dtick=5)
    fig.update_yaxes(autorange="reversed")
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def plot_retrieved_safe_cited_tranco_ranks():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_tranco_ranks.pkl"
    ).copy()
    plot_rows = _build_retrieved_safe_cited_rank_plot_rows(
        df,
        retrieved_rank_col="ranks_srcs_retrieved",
        cited_rank_col="ranks_srcs_cited",
        min_rank=0,
    )
    _plot_retrieved_safe_cited_ranks(
        plot_rows,
        file_name="retrieved_safe_cited_tranco_ranks",
        yaxis_title="Tranco Rank",
    )


def plot_retrieved_safe_cited_judge_ranks():
    df = _load_first_existing_pickle(
        [
            f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks.pkl",
            f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks_v2.pkl",
        ]
    )
    plot_rows = _build_retrieved_safe_cited_rank_plot_rows(
        df,
        retrieved_rank_col="reliability_scores_srcs_retrieved",
        cited_rank_col="reliability_scores_srcs_cited",
        rank_transform=lambda score: 5 - score,
    )
    _plot_retrieved_safe_cited_ranks(
        plot_rows,
        file_name="retrieved_safe_cited_judge_ranks",
        yaxis_title="Judge Rank (5 - Score)",
    )


def cited_sources_reachability():
    cache_path = f"{OUTPUT_PATH}/metadata/cited_sources_reachability.csv"
    if os.path.exists(cache_path):
        reachability_df = pd.read_csv(cache_path)
    else:
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
        ).copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

        unique_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            retrieved_urls = {
                item.get("url", "")
                for item in row["srcs_retrieved"]
                if isinstance(item, dict) and item.get("url", "")
            }
            retrieved_domains = {
                item.get("domain", "")
                for item in row["srcs_retrieved"]
                if isinstance(item, dict) and item.get("domain", "")
            }
            cited_items = [
                item
                for item in row["srcs_cited"]
                if isinstance(item, dict) and item.get("url", "")
            ]

            novel_cited_urls = sorted(
                {
                    item["url"]
                    for item in cited_items
                    if item["url"] not in retrieved_urls
                    and item.get("domain", "") not in retrieved_domains
                }
            )
            reachable_novel_cited = count_reachable_urls(novel_cited_urls)

            unique_rows.append(
                {
                    "month": row["month"],
                    "novel_cited_urls": novel_cited_urls,
                    "num_novel_cited_urls": len(novel_cited_urls),
                    "num_novel_cited_urls_hallucinated": len(novel_cited_urls) - reachable_novel_cited,
                }
            )

        reachability_df = pd.DataFrame(unique_rows)
        reachability_df.to_csv(cache_path, index=False)

    plot_hallucination_rate_over_time(
        reachability_df,
        total_col="num_novel_cited_urls",
        hallucinated_col="num_novel_cited_urls_hallucinated",
        file_name="novel_cited_url_hallucination_rate_over_time",
        yaxis_title="Hallucinated Cited URLs (%)",
    )


def plot_citations_round():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()

    round_counts = {}
    for _, row in df.iterrows():
        cited_turns = [
            item.get("turn_index")
            for item in row["srcs_cited"]
            if isinstance(item, dict) and item.get("turn_index") is not None
        ]
        for cited_turn in cited_turns:
            round_counts[cited_turn] = round_counts.get(cited_turn, 0) + 1

    if not round_counts:
        return

    plot_df = (
        pd.DataFrame(
            {
                "round": list(round_counts.keys()),
                "count": list(round_counts.values()),
            }
        )
        .sort_values("round")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["round"],
            y=plot_df["count"],
            text=plot_df["count"],
            textposition="auto",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Citation Round",
        yaxis_title="Number of Citations",
    )
    file_name = "citation_round_distribution"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def check_url(url):
    if not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False

    session = requests.Session()
    session.headers.update(HEADERS)

    existing_statuses = {200, 201, 202, 204, 206, 301, 302, 303, 307, 308, 401, 403, 405, 406, 429}

    try:
        response = session.head(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
        )
        if response.status_code in existing_statuses:
            return True
    except requests.RequestException:
        pass

    try:
        response = session.get(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
            stream=True,
            headers={**HEADERS, "Range": "bytes=0-0"},
        )
        return response.status_code in existing_statuses
    except requests.RequestException:
        return False
    finally:
        session.close()
    

    # # Step 1: Format check
    # parsed = urlparse(url)
    # if parsed.scheme not in ("http", "https") or not parsed.netloc:
    #     return False

    # host = parsed.netloc

    # # Step 2: DNS Resolution
    # try:
    #     ip = socket.gethostbyname(host)
    # except Exception:
    #     return False

    # # Step 3: TCP Connection
    # port = 443 if parsed.scheme == "https" else 80
    # try:
    #     sock = socket.create_connection((host, port), timeout=TIMEOUT)
    # except Exception:
    #     return False

    # # Step 4: SSL Handshake (HTTPS only)
    # if parsed.scheme == "https":
    #     try:
    #         context = ssl.create_default_context()
    #         sock = context.wrap_socket(sock, server_hostname=host)
    #     except Exception:
    #         return False

    # # Step 5: HTTP Request (with browser headers)
    # try:
    #     response = requests.get(
    #         url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True
    #     )
    #     return True
    # except requests.exceptions.Timeout:
    #     return False
    # except requests.exceptions.RequestException:
    #     return False


def count_reachable_urls(urls):
    urls = sorted(set(urls))
    if not urls:
        return 0

    with ThreadPoolExecutor(max_workers=min(100, len(urls))) as executor:
        return sum(int(result) for result in executor.map(check_url, urls))


def plot_hallucination_rate_over_time(df, total_col, hallucinated_col, file_name, yaxis_title):
    plot_df = df.copy()
    plot_df["month"] = pd.to_datetime(plot_df["month"], errors="coerce")
    plot_df = plot_df.dropna(subset=["month"])

    monthly = (
        plot_df.groupby("month")[[total_col, hallucinated_col]]
        .sum()
        .sort_index()
    )
    if len(monthly) == 0:
        return

    full_month_range = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS",
    )
    monthly = monthly.reindex(full_month_range, fill_value=0).rename_axis("month").reset_index()
    monthly["hallucination_rate"] = (
        monthly[hallucinated_col] / monthly[total_col].replace(0, pd.NA)
    )
    monthly["hallucination_rate"] = monthly["hallucination_rate"].fillna(0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["hallucination_rate"],
            mode="lines+markers",
            connectgaps=True,
            marker=dict(size=7),
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title=yaxis_title,
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%")
    fig.add_vline(
        x=pd.Timestamp("2024-11-01"),
        line_width=2,
        line_dash="dash",
        line_color="black",
    )
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 17))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def venn_diagram_of_sources():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    )

    retrieved_urls = [
        _normalize_url_for_source_matching(src.get("url", ""))
        for sources in df["srcs_retrieved"]
        for src in sources
        if isinstance(src, dict) and src.get("url")
    ]
    cited_urls = [
        _normalize_url_for_source_matching(src.get("url", ""))
        for sources in df["srcs_cited"]
        for src in sources
        if isinstance(src, dict) and src.get("url")
    ]
    print(len(retrieved_urls))
    print(len(cited_urls))
    cited_external = [x for x in cited_urls if x in retrieved_urls]
    print(len(cited_external))


    retrieved_urls = {
        _normalize_url_for_source_matching(src.get("url", ""))
        for sources in df["srcs_retrieved"]
        for src in sources
        if isinstance(src, dict) and src.get("url")
    }
    cited_urls = {
        _normalize_url_for_source_matching(src.get("url", ""))
        for sources in df["srcs_cited"]
        for src in sources
        if isinstance(src, dict) and src.get("url")
    }
    retrieved_set = {url for url in retrieved_urls if url}
    cited_set = {url for url in cited_urls if url}

    cited_external = cited_set & retrieved_set
    cited_internal = cited_set - retrieved_set

    print(len(retrieved_set))
    print(len(cited_set))
    print(len(cited_external))

    cited_external = 133264
    cited_internal = 17796
    external_valid = 127952
    external_non_valid = 5312
    internal_valid = 17198
    internal_non_valid = 598

    fig = go.Figure()

    outer_center_x = 2.0
    outer_center_y = 2.0
    outer_radius = 1.35
    inner_radius = 0.78

    def _disk_sector_points(cx, cy, radius, theta_start, theta_end, n_points=280):
        theta = np.linspace(theta_start, theta_end, n_points)
        x_arc = cx + radius * np.cos(theta)
        y_arc = cy + radius * np.sin(theta)
        return np.concatenate(([cx], x_arc, [cx])), np.concatenate(([cy], y_arc, [cy]))

    def _annulus_sector_points(
        cx,
        cy,
        radius_outer,
        radius_inner,
        theta_start,
        theta_end,
        n_points=280,
    ):
        theta_outer = np.linspace(theta_start, theta_end, n_points)
        theta_inner = np.linspace(theta_end, theta_start, n_points)

        x_outer = cx + radius_outer * np.cos(theta_outer)
        y_outer = cy + radius_outer * np.sin(theta_outer)
        x_inner = cx + radius_inner * np.cos(theta_inner)
        y_inner = cy + radius_inner * np.sin(theta_inner)
        return np.concatenate([x_outer, x_inner]), np.concatenate([y_outer, y_inner])

    inner_total = external_valid + external_non_valid
    ring_total = internal_valid + internal_non_valid
    inner_valid_fraction = external_valid / inner_total if inner_total > 0 else 0.5
    ring_valid_fraction = internal_valid / ring_total if ring_total > 0 else 0.5

    theta_start = np.pi / 2
    theta_full = theta_start + 2 * np.pi
    theta_inner_split = theta_start + 2 * np.pi * inner_valid_fraction
    theta_ring_split = theta_start + 2 * np.pi * ring_valid_fraction

    region_specs = [
        {
            "name": "Retrieved / External (Valid)",
            "region_type": "inner",
            "color": "#3765E5",
            "count": external_valid,
            "theta0": theta_start,
            "theta1": theta_inner_split,
        },
        {
            "name": "Retrieved / External (Non-valid)",
            "region_type": "inner",
            "color": "#9DB8FF",
            "count": external_non_valid,
            "theta0": theta_inner_split,
            "theta1": theta_full,
        },
        {
            "name": "Internal / Unexplained (Valid)",
            "region_type": "ring",
            "color": "#E45756",
            "count": internal_valid,
            "theta0": theta_start,
            "theta1": theta_ring_split,
        },
        {
            "name": "Internal / Unexplained (Non-valid)",
            "region_type": "ring",
            "color": "#FFB3A7",
            "count": internal_non_valid,
            "theta0": theta_ring_split,
            "theta1": theta_full,
        },
    ]

    count_annotations = []
    for spec in region_specs:
        if spec["region_type"] == "inner":
            x_points, y_points = _disk_sector_points(
                outer_center_x,
                outer_center_y,
                inner_radius,
                spec["theta0"],
                spec["theta1"],
            )
            label_radius = inner_radius * 0.58
        else:
            x_points, y_points = _annulus_sector_points(
                outer_center_x,
                outer_center_y,
                outer_radius,
                inner_radius,
                spec["theta0"],
                spec["theta1"],
            )
            label_radius = (outer_radius + inner_radius) / 2

        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=y_points,
                mode="lines",
                fill="toself",
                line=dict(color="rgba(0,0,0,0.35)", width=1.5),
                fillcolor=spec["color"],
                name=spec["name"],
                hovertemplate=(
                    f"{spec['name']}<br>Count: {spec['count']}<extra></extra>"
                ),
            )
        )
        mid_theta = 0.5 * (spec["theta0"] + spec["theta1"])
        count_annotations.append(
            (
                outer_center_x + label_radius * np.cos(mid_theta),
                outer_center_y + label_radius * np.sin(mid_theta),
                str(spec["count"]),
            )
        )

    fig.add_shape(
        type="circle",
        x0=outer_center_x - outer_radius,
        y0=outer_center_y - outer_radius,
        x1=outer_center_x + outer_radius,
        y1=outer_center_y + outer_radius,
        line=dict(color="rgba(0,0,0,0.7)", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.add_shape(
        type="circle",
        x0=outer_center_x - inner_radius,
        y0=outer_center_y - inner_radius,
        x1=outer_center_x + inner_radius,
        y1=outer_center_y + inner_radius,
        line=dict(color="rgba(0,0,0,0.7)", width=2),
        fillcolor="rgba(0,0,0,0)",
    )

    for x, y, text in count_annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=21, color="black"),
        )

    label_annotations = [
        (outer_center_x, outer_center_y + outer_radius + 0.18, "Cited URLs"),
    ]
    for x, y, text in label_annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=16, color="black"),
        )

    fig.update_layout(
        xaxis=dict(visible=False, range=[0.58, 3.38]),
        yaxis=dict(visible=False, range=[0.58, 3.66], scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=26, pad=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    file_name = "venn_diagram_of_sources"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(
        fig,
        config=styler(18, 16),
        new_legend=dict(
            x=0.5,
            y=-0.03,
            xanchor="center",
            yanchor="top",
            orientation="h",
            entrywidthmode="fraction",
            entrywidth=0.48,
            traceorder="normal",
            bgcolor="rgba(255,255,255,0.75)",
            font=dict(size=14, color="black"),
        ),
    )
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def evaluate_source_tranco_ranks(separate_cited_external_internal=False):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_tranco_ranks.pkl"
    ).copy()

    def _as_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return []
        return []

    def _avg_valid_rank(ranks):
        ranks = _as_list(ranks)
        valid_ranks = []
        for rank in ranks:
            try:
                rank_value = float(rank)
            except (TypeError, ValueError):
                continue
            if rank_value > 0:
                valid_ranks.append(rank_value)
        if not valid_ranks:
            return np.nan
        return float(np.mean(valid_ranks))

    def _avg_valid_rank_by_mask(ranks, keep_mask):
        ranks = _as_list(ranks)
        valid_ranks = []
        for idx, rank in enumerate(ranks):
            if idx >= len(keep_mask) or not keep_mask[idx]:
                continue
            try:
                rank_value = float(rank)
            except (TypeError, ValueError):
                continue
            if rank_value > 0:
                valid_ranks.append(rank_value)
        if not valid_ranks:
            return np.nan
        return float(np.mean(valid_ranks))

    df["retrieved_avg_rank"] = df["ranks_srcs_retrieved"].apply(_avg_valid_rank)

    if separate_cited_external_internal:
        cited_external_avg = []
        cited_internal_avg = []
        for _, row in df.iterrows():
            retrieved_urls = {
                _normalize_url_for_source_matching(item.get("url", ""))
                for item in _as_list(row.get("srcs_retrieved", []))
                if isinstance(item, dict) and item.get("url", "")
            }
            cited_sources = _as_list(row.get("srcs_cited", []))
            external_mask = []
            internal_mask = []
            for item in cited_sources:
                if not isinstance(item, dict):
                    external_mask.append(False)
                    internal_mask.append(False)
                    continue
                cited_url = _normalize_url_for_source_matching(item.get("url", ""))
                is_external = bool(cited_url and cited_url in retrieved_urls)
                external_mask.append(is_external)
                internal_mask.append(not is_external)

            cited_external_avg.append(
                _avg_valid_rank_by_mask(row.get("ranks_srcs_cited", []), external_mask)
            )
            cited_internal_avg.append(
                _avg_valid_rank_by_mask(row.get("ranks_srcs_cited", []), internal_mask)
            )

        df["cited_external_avg_rank"] = cited_external_avg
        df["cited_internal_avg_rank"] = cited_internal_avg
        rank_specs = [
            ("retrieved_avg_rank", "Retrieved", "#636EFA"),
            ("cited_external_avg_rank", "Cited<br>Retrieved", "#00CC96"),
            ("cited_internal_avg_rank", "Cited<br>Parametric", "#E45756"),
        ]
    else:
        df["cited_avg_rank"] = df["ranks_srcs_cited"].apply(_avg_valid_rank)
        rank_specs = [
            ("retrieved_avg_rank", "Retrieved", "#636EFA"),
            ("cited_avg_rank", "Cited", "#EF553B"),
        ]

    print("Average Tranco rank ranges:")
    for col, label, _color in rank_specs:
        subset = df[col].dropna()
        if len(subset) == 0:
            print(f"{label}: no valid ranks")
        else:
            print(f"{label}: {subset.min():,.0f}-{subset.max():,.0f}")

    box_df = df[[col for col, _label, _color in rank_specs]].copy()
    box_fig = go.Figure()

    all_positive_values = []
    for col, _label, _color in rank_specs:
        col_values = pd.to_numeric(box_df[col], errors="coerce").to_numpy()
        col_values = col_values[np.isfinite(col_values)]
        col_values = col_values[col_values > 0]
        if len(col_values) > 0:
            all_positive_values.append(col_values)

    if len(all_positive_values) > 0:
        combined_values = np.concatenate(all_positive_values)
        global_min_log = float(np.floor(np.log10(np.min(combined_values))))
        global_max_log = float(np.ceil(np.log10(np.max(combined_values))))
        if global_max_log <= global_min_log:
            global_max_log = global_min_log + 1.0
    else:
        global_min_log = 0.0
        global_max_log = 1.0

    for col, label, color in rank_specs:
        subset = box_df[col].dropna()
        subset = subset[subset > 0]
        if len(subset) == 0:
            continue
        subset_log = np.log10(subset.to_numpy())
        box_fig.add_trace(
            go.Violin(
                x=[label] * len(subset_log),
                y=subset_log,
                name=label,
                marker_color=color,
                line_color=color,
                width=0.9,
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                legendgroup=label,
            )
        )

    box_fig.update_layout(
        xaxis_title="Source Type",
        xaxis=dict(tickangle=0),
        yaxis_title="Average Rank (log10)",
        yaxis=dict(range=[global_min_log, global_max_log], tickmode="linear", dtick=1),
        violinmode="group",
    )
    violin_file_name = "source_rank_violinplot"
    if separate_cited_external_internal:
        violin_file_name += "_split_cited"
    box_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{violin_file_name}.html")
    box_fig = with_paper_style(box_fig, config=styler(20, 16), legend_pos=None)
    box_fig.update_xaxes(tickangle=0, tickfont=dict(size=20))
    box_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{violin_file_name}.pdf", format="pdf")

    paired_fig = make_subplots(rows=1, cols=1)
    if separate_cited_external_internal:
        scatter_specs = [
            ("cited_external_avg_rank", "Cited Retrieved", "#00CC96"),
            ("cited_internal_avg_rank", "Cited Parametric", "#E45756"),
        ]
    else:
        scatter_specs = [
            ("cited_avg_rank", "Cited", "#00CC96"),
        ]

    diagonal_min = np.inf
    diagonal_max = -np.inf
    has_points = False
    for y_col, label, color in scatter_specs:
        subset = df[["retrieved_avg_rank", y_col]].dropna()
        if len(subset) == 0:
            continue
        has_points = True
        diagonal_min = min(
            diagonal_min,
            float(subset["retrieved_avg_rank"].min()),
            float(subset[y_col].min()),
        )
        diagonal_max = max(
            diagonal_max,
            float(subset["retrieved_avg_rank"].max()),
            float(subset[y_col].max()),
        )
        paired_fig.add_trace(
            go.Scatter(
                x=subset["retrieved_avg_rank"],
                y=subset[y_col],
                mode="markers",
                name=label,
                marker=dict(color=color, size=8),
                showlegend=separate_cited_external_internal,
            ),
            row=1,
            col=1,
        )

    if has_points:
        if diagonal_min == diagonal_max:
            diagonal_min -= 1
            diagonal_max += 1
        paired_fig.add_trace(
            go.Scatter(
                x=[diagonal_min, diagonal_max],
                y=[diagonal_min, diagonal_max],
                mode="lines",
                line=dict(color="black", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        paired_fig.update_xaxes(
            title_text="Retrieved Avg Rank",
            range=[diagonal_min, diagonal_max],
            row=1,
            col=1,
        )
        y_title = "Cited Avg Rank"
        if separate_cited_external_internal:
            y_title = "Cited Avg Rank (External/Internal)"
        paired_fig.update_yaxes(
            title_text=y_title,
            range=[diagonal_min, diagonal_max],
            row=1,
            col=1,
        )

    paired_file_name = "source_rank_paired_plot"
    if separate_cited_external_internal:
        paired_file_name += "_split_cited"
    paired_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{paired_file_name}.html")
    legend_pos = (0.98, 1.2) if separate_cited_external_internal else None
    paired_fig = with_paper_style(
        paired_fig,
        config=styler(18, 16),
        legend_pos=legend_pos,
    )
    paired_fig.update_layout(width=700 if separate_cited_external_internal else 500, height=400)
    paired_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{paired_file_name}.pdf", format="pdf")


def evaluate_source_topical_judge_ranks():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks.pkl"
    ).copy()

    def _as_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return []
        return []

    def _avg_valid_score(scores):
        scores = _as_list(scores)
        valid_scores = []
        for score in scores:
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            if np.isfinite(score_value):
                valid_scores.append(score_value)
        if not valid_scores:
            return np.nan
        return float(np.mean(valid_scores))

    source_score_specs = [
        ("retrieved_topic_reliability", "Retrieved", "reliability_scores_srcs_retrieved", "#636EFA"),
        ("cited_topic_reliability", "Cited", "reliability_scores_srcs_cited", "#EF553B"),
    ]
    for avg_col, _label, score_col, _color in source_score_specs:
        if score_col and score_col in df.columns:
            df[avg_col] = df[score_col].apply(_avg_valid_score)
        else:
            df[avg_col] = np.nan

    print("Average topical judge score ranges:")
    for avg_col, label, _score_col, _color in source_score_specs:
        subset = df[avg_col].dropna()
        if len(subset) == 0:
            print(f"{label}: no valid scores")
        else:
            print(f"{label}: {subset.min():.2f}-{subset.max():.2f}")

    score_df = df[[avg_col for avg_col, _label, _score_col, _color in source_score_specs]].copy()

    violin_fig = go.Figure()
    for avg_col, label, _score_col, color in source_score_specs:
        subset = score_df[avg_col].dropna()
        if len(subset) == 0:
            violin_fig.add_trace(
                go.Scatter(
                    x=[label],
                    y=[None],
                    mode="markers",
                    marker=dict(opacity=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            continue
        violin_fig.add_trace(
            go.Violin(
                x=[label] * len(subset),
                y=subset,
                name=label,
                marker_color=color,
                line_color=color,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            )
        )

    violin_fig.update_layout(
        xaxis_title="Source Type",
        yaxis_title="Average Score",
        xaxis=dict(
            categoryorder="array",
            categoryarray=["Retrieved", "Cited"],
        ),
        # height=600,
        # width=800,
    )
    violin_fig.update_yaxes(range=[1, 5])
    file_name = "source_topic_judge_rank_violinplot"
    violin_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    violin_fig = with_paper_style(violin_fig, config=styler(18, 16), legend_pos=None)
    # violin_fig.update_layout(height=600, width=800)
    violin_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    paired_fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["Retrieved vs Cited"],
    )
    paired_specs = [
        ("retrieved_topic_reliability", "cited_topic_reliability", "Retrieved vs Cited", "#00CC96"),
    ]
    for idx, (x_col, y_col, label, color) in enumerate(paired_specs, start=1):
        subset = df[[x_col, y_col]].dropna()
        if len(subset) > 0:
            diagonal_min = min(subset[x_col].min(), subset[y_col].min())
            diagonal_max = max(subset[x_col].max(), subset[y_col].max())
            if diagonal_min == diagonal_max:
                diagonal_min -= 0.5
                diagonal_max += 0.5
            on_line = pd.Series(
                np.isclose(subset[y_col], subset[x_col]),
                index=subset.index,
            )
            above_line = subset[y_col] > subset[x_col]
            below_line = subset[y_col] < subset[x_col]
            above_count = int((above_line & ~on_line).sum())
            below_count = int((below_line & ~on_line).sum())
            on_line_count = int(on_line.sum())
            paired_fig.add_trace(
                go.Scatter(
                    x=subset[x_col],
                    y=subset[y_col],
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=8),
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )
            paired_fig.add_trace(
                go.Scatter(
                    x=[diagonal_min, diagonal_max],
                    y=[diagonal_min, diagonal_max],
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )
            paired_fig.update_xaxes(range=[diagonal_min, diagonal_max], row=1, col=idx)
            paired_fig.update_yaxes(range=[diagonal_min, diagonal_max], row=1, col=idx)
        else:
            above_count = 0
            below_count = 0
            on_line_count = 0
            paired_fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{'' if idx == 1 else idx} domain",
                yref=f"y{'' if idx == 1 else idx} domain",
                text="No paired data",
                showarrow=False,
                font=dict(size=14, color="gray"),
            )
            paired_fig.update_xaxes(range=[1, 5], row=1, col=idx)
            paired_fig.update_yaxes(range=[1, 5], row=1, col=idx)

        axis_suffix = "" if idx == 1 else str(idx)
        paired_fig.add_annotation(
            x=0.04,
            y=0.95,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"{above_count} points",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=13, color=color),
            bgcolor="rgba(255,255,255,0.78)",
        )
        paired_fig.add_annotation(
            x=0.54,
            y=0.54,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"{on_line_count} points",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=13, color="black"),
            bgcolor="rgba(255,255,255,0.78)",
        )
        paired_fig.add_annotation(
            x=0.96,
            y=0.05,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"{below_count} points",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=13, color=color),
            bgcolor="rgba(255,255,255,0.78)",
        )
        paired_fig.update_xaxes(title_text=x_col.replace("_", " ").title(), row=1, col=idx)
        paired_fig.update_yaxes(title_text=y_col.replace("_", " ").title(), row=1, col=idx)

    file_name = "source_topic_judge_rank_paired_plot"
    paired_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    paired_fig = with_paper_style(paired_fig, config=styler(18, 16), legend_pos=None)
    paired_fig.update_layout(width=500, height=400)
    paired_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def calculate_invivo_tranco_rank_correlation():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_tranco_ranks.pkl"
    ).copy()

    tranco_col = "ranks_srcs_retrieved"
    invivo_col = "srcs_retrieved"
    missing_cols = [col for col in [tranco_col, invivo_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    id_cols = [col for col in ["user_id", "conv_id", "turn_id", "topic"] if col in df.columns]

    pair_rows = []
    sample_rows = []
    for sample_index, (_, row) in enumerate(df.iterrows()):
        row_id = {col: row[col] for col in id_cols}
        row_id["sample_index"] = int(sample_index)

        tranco_values = _as_list(row.get(tranco_col, []))
        invivo_items = _as_list(row.get(invivo_col, []))
        paired_count = min(len(tranco_values), len(invivo_items))
        if paired_count == 0:
            continue

        sample_tranco_ranks = []
        sample_invivo_ranks = []
        for idx in range(paired_count):
            tranco_rank = _as_valid_number(tranco_values[idx], min_value=1)
            if not np.isfinite(tranco_rank):
                continue

            invivo_item = invivo_items[idx]
            invivo_rank = np.nan
            if isinstance(invivo_item, dict):
                invivo_rank = _as_valid_number(invivo_item.get("ref_index"), min_value=0)
                if np.isfinite(invivo_rank):
                    invivo_rank += 1.0
            if not np.isfinite(invivo_rank):
                invivo_rank = float(idx + 1)

            sample_tranco_ranks.append(float(tranco_rank))
            sample_invivo_ranks.append(float(invivo_rank))
            pair_rows.append(
                {
                    **row_id,
                    "source_index": int(idx),
                    "tranco_rank": float(tranco_rank),
                    "invivo_rank": float(invivo_rank),
                }
            )

        if len(sample_tranco_ranks) == 0:
            continue
        sample_rows.append(
            {
                **row_id,
                "num_pairs": int(len(sample_tranco_ranks)),
                "avg_tranco_rank": float(np.mean(sample_tranco_ranks)),
                "avg_invivo_rank": float(np.mean(sample_invivo_ranks)),
            }
        )

    pair_columns = id_cols + [
        "sample_index",
        "source_index",
        "tranco_rank",
        "invivo_rank",
    ]
    pair_df = pd.DataFrame(pair_rows, columns=pair_columns)
    sample_columns = id_cols + [
        "sample_index",
        "num_pairs",
        "avg_tranco_rank",
        "avg_invivo_rank",
    ]
    sample_avg_df = pd.DataFrame(sample_rows, columns=sample_columns)
    if len(pair_df) == 0:
        raise ValueError("No valid Tranco/InVivo rank pairs found for plotting.")
    if len(sample_avg_df) == 0:
        raise ValueError("No valid per-sample averages found for plotting.")

    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    pair_df.to_csv(f"{output_dir}/invivo_tranco_rank_pairs.csv", index=False)
    sample_avg_df.to_csv(
        f"{output_dir}/invivo_tranco_rank_per_sample_avg.csv",
        index=False,
    )

    corr_pair_rows = []
    corr_sample_rows = []
    for sample_index, sample_df in pair_df.groupby("sample_index", sort=False):
        sample_meta = sample_df.iloc[0].to_dict()
        row_id = {col: sample_meta.get(col) for col in id_cols}
        row_id["sample_index"] = int(sample_index)

        # Keep only valid Tranco ranks (explicitly excluding -1).
        sample_valid = sample_df[pd.to_numeric(sample_df["tranco_rank"], errors="coerce") != -1].copy()
        sample_valid = sample_valid.dropna(subset=["tranco_rank", "invivo_rank"])

        if len(sample_valid) == 0:
            corr_sample_rows.append(
                {
                    **row_id,
                    "num_pairs_valid_tranco": 0,
                    "rank_corr_spearman": None,
                    "rank_corr_pearson": None,
                    "exact_order_match": None,
                }
            )
            continue

        sample_valid["tranco_order_rank"] = sample_valid["tranco_rank"].rank(
            method="first",
            ascending=True,
        )
        sample_valid["invivo_order_rank"] = sample_valid["invivo_rank"].rank(
            method="first",
            ascending=True,
        )

        for _, row in sample_valid.iterrows():
            corr_pair_rows.append(
                {
                    **row_id,
                    "source_index": int(row["source_index"]),
                    "tranco_rank": float(row["tranco_rank"]),
                    "invivo_rank": float(row["invivo_rank"]),
                    "tranco_order_rank": float(row["tranco_order_rank"]),
                    "invivo_order_rank": float(row["invivo_order_rank"]),
                }
            )

        if len(sample_valid) >= 2:
            rank_corr_spearman = sample_valid["tranco_order_rank"].corr(
                sample_valid["invivo_order_rank"],
                method="spearman",
            )
            rank_corr_pearson = sample_valid["tranco_order_rank"].corr(
                sample_valid["invivo_order_rank"],
                method="pearson",
            )
            tranco_order = tuple(
                sample_valid.sort_values(
                    ["tranco_order_rank", "source_index"],
                    ascending=[True, True],
                )["source_index"].tolist()
            )
            invivo_order = tuple(
                sample_valid.sort_values(
                    ["invivo_order_rank", "source_index"],
                    ascending=[True, True],
                )["source_index"].tolist()
            )
            exact_order_match = bool(tranco_order == invivo_order)
        else:
            rank_corr_spearman = None
            rank_corr_pearson = None
            exact_order_match = None

        corr_sample_rows.append(
            {
                **row_id,
                "num_pairs_valid_tranco": int(len(sample_valid)),
                "rank_corr_spearman": (
                    float(rank_corr_spearman) if pd.notna(rank_corr_spearman) else None
                ),
                "rank_corr_pearson": (
                    float(rank_corr_pearson) if pd.notna(rank_corr_pearson) else None
                ),
                "exact_order_match": exact_order_match,
            }
        )

    corr_pair_columns = id_cols + [
        "sample_index",
        "source_index",
        "tranco_rank",
        "invivo_rank",
        "tranco_order_rank",
        "invivo_order_rank",
    ]
    corr_sample_columns = id_cols + [
        "sample_index",
        "num_pairs_valid_tranco",
        "rank_corr_spearman",
        "rank_corr_pearson",
        "exact_order_match",
    ]
    corr_pair_df = pd.DataFrame(corr_pair_rows, columns=corr_pair_columns)
    corr_sample_df = pd.DataFrame(corr_sample_rows, columns=corr_sample_columns)
    corr_pair_df.to_csv(
        f"{output_dir}/invivo_tranco_rank_pairs_with_order_valid_tranco.csv",
        index=False,
    )
    corr_sample_df.to_csv(
        f"{output_dir}/invivo_tranco_rank_correlation_per_sample.csv",
        index=False,
    )

    corr_known = corr_sample_df.dropna(subset=["rank_corr_spearman"]) if len(corr_sample_df) > 0 else corr_sample_df
    exact_known = corr_sample_df.dropna(subset=["exact_order_match"]) if len(corr_sample_df) > 0 else corr_sample_df
    exact_order_match_rate = (
        float(pd.to_numeric(exact_known["exact_order_match"], errors="coerce").mean())
        if len(exact_known) > 0
        else None
    )
    corr_summary = {
        "method": (
            "Per-sample ranking correlation between Tranco and InVivo ranks "
            "over retrieved URLs with valid Tranco (Tranco != -1)"
        ),
        "n_pairs_valid_tranco": int(len(corr_pair_df)),
        "n_samples_total": int(len(corr_sample_df)),
        "n_samples_with_correlation": int(len(corr_known)),
        "mean_rank_corr_spearman": (
            float(pd.to_numeric(corr_known["rank_corr_spearman"], errors="coerce").mean())
            if len(corr_known) > 0
            else None
        ),
        "median_rank_corr_spearman": (
            float(pd.to_numeric(corr_known["rank_corr_spearman"], errors="coerce").median())
            if len(corr_known) > 0
            else None
        ),
        "mean_rank_corr_pearson": (
            float(pd.to_numeric(corr_known["rank_corr_pearson"], errors="coerce").mean())
            if len(corr_known) > 0
            else None
        ),
        "exact_order_match_rate": exact_order_match_rate,
    }
    to_json(corr_summary, f"{output_dir}/invivo_tranco_rank_correlation_summary.json")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_avg_df["avg_tranco_rank"],
            y=sample_avg_df["avg_invivo_rank"],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=6,
                opacity=0.45,
                color="#636EFA",
            ),
            hovertemplate=(
                "Avg Tranco Rank=%{x:.2f}<br>"
                "Avg InVivo Rank=%{y:.2f}<br>"
                "Pairs in sample=%{customdata}<extra></extra>"
            ),
            customdata=sample_avg_df["num_pairs"],
        )
    )
    fig.update_layout(
        xaxis_title="Average Tranco Rank (Per Sample)",
        yaxis_title="Average InVivo Rank (Per Sample)",
    )

    file_name = "invivo_tranco_rank_scatter"
    fig.write_html(f"{output_dir}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16), legend_pos=None)
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

    summary = {
        "n_source_pairs": int(len(pair_df)),
        "n_samples": int(len(sample_avg_df)),
        "max_tranco_rank": float(pair_df["tranco_rank"].max()),
        "max_invivo_rank": float(pair_df["invivo_rank"].max()),
        "rank_correlation": corr_summary,
    }
    print(json.dumps(summary, indent=2))
    return sample_avg_df


def add_retrieved_safe_reliability_scores_to_topical_judge_ranks():
    df = _load_response_source_similarity_input()
    df_scores = pd.read_csv(
        f"{OUTPUT_PATH}/metadata/source_reliability_scores.csv"
    )

    required_score_cols = {"topic", "url", "score"}
    missing_score_cols = required_score_cols - set(df_scores.columns)
    if missing_score_cols:
        raise ValueError(
            f"Missing required score columns: {sorted(missing_score_cols)}"
        )

    source_specs = [
        ("srcs_retrieved", "reliability_scores_srcs_retrieved"),
        ("srcs_safe_urls", "reliability_scores_srcs_safe"),
        ("srcs_cited", "reliability_scores_srcs_cited"),
    ]
    missing_source_cols = [
        source_col
        for source_col, _score_col in source_specs
        if source_col not in df.columns
    ]
    if missing_source_cols:
        raise ValueError(
            f"Missing required source columns: {sorted(missing_source_cols)}"
        )

    def _as_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return []
            return parsed if isinstance(parsed, list) else []
        return []

    def _normalize_topic(value):
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    def _normalize_url(value):
        if value is None or pd.isna(value):
            return ""
        return (
            str(value)
            .strip()
            .removesuffix("?utm_source=chatgpt.com")
            .removesuffix("&utm_source=chatgpt.com")
        )

    def _build_lookup(score_df):
        lookup = {}
        for _, score_row in score_df.iterrows():
            topic = _normalize_topic(score_row.get("topic", ""))
            url = _normalize_url(score_row.get("url", ""))
            if not topic or not url:
                continue
            try:
                score = float(score_row.get("score"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score):
                continue
            lookup[(topic, url)] = score
        return lookup

    score_lookup = _build_lookup(df_scores)

    def _score_sources(row, source_col):
        topic = _normalize_topic(row.get("topic", ""))
        scores = []
        for src in _as_list(row.get(source_col, [])):
            if isinstance(src, dict):
                url = _normalize_url(src.get("url", ""))
            else:
                url = _normalize_url(src)
            scores.append(score_lookup.get((topic, url), np.nan))
        return scores

    score_columns = {score_col: [] for _source_col, score_col in source_specs}
    for _, row in df.iterrows():
        for source_col, score_col in source_specs:
            score_columns[score_col].append(_score_sources(row, source_col))

    for score_col, scores in score_columns.items():
        df[score_col] = scores

    df.to_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks.pkl"
    )
    df.to_csv(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks.csv",
        index=False,
    )
    print(f"Final length: {len(df)}")


if __name__ == "__main__":
    # web_df = load_web_data_from_file(fmt="pkl")
    # print(f"Loaded web data: {len(web_df)}")
    # extract_retrieved_safe_cited_source(web_df)
    
    # print(count_unique_retrieved_safe_cited())
    # save_topic_to_domains_json()

    # plot_subset_condition_counts()

    # plot_url_counts_over_time(separate_cited_external_internal=True)
    # plot_grounding_rate_violin_over_time()
    # plot_retrieved_url_counts_over_time_by_model()
    # plot_url_counts_by_model()
    # plot_url_counts_by_topic()

    # plot_retrieved_safe_cited_positions(separate_cited_external_internal=True)
    # plot_citations_round()

    # evaluate_source_tranco_ranks(separate_cited_external_internal=True)
    # evaluate_source_topical_judge_ranks()

    # plot_top_domains(separate_cited_external_internal=True)
    # plot_top_domains_by_selected_topics(separate_cited_external_internal=True)
    # plot_top_domains_by_model(separate_cited_external_internal=True)

    venn_diagram_of_sources()

    # calculate_invivo_tranco_rank_correlation()
    pass
