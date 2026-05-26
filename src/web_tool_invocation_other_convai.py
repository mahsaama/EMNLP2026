import argparse
import os
import sys
import csv
import ast
import json
from collections import Counter
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import qualitative
# from utils import *
# from data_utils import *
from paper import with_paper_style, styler
from data_extraction import load_web_data_from_file, load_whole_data_from_file
from utils import load_json, to_json

CONF = "emnlp/web_tool_invocation"
OUTPUT_PATH = "./outputs/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="chatgpt")
    return parser.parse_args()

def _primary_model(models):
    if not isinstance(models, list):
        return "Unknown"
    cleaned = [model for model in models if isinstance(model, str) and model]
    if not cleaned:
        return "Unknown"
    return cleaned[-1]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _confusion_counts(y_true, y_pred):
    counts = [[0, 0], [0, 0]]
    for truth, pred in zip(y_true, y_pred):
        truth_idx = 1 if truth else 0
        pred_idx = 1 if pred else 0
        counts[truth_idx][pred_idx] += 1
    return counts


def _confusion_annotations(counts, total, row_labels, col_labels, cell_text=None):
    annotations = []
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            count = counts[row_idx][col_idx]
            share = count / total if total else 0
            text = f"{count}<br>{share:.0%}"
            if cell_text is not None:
                text = f"{cell_text[row_idx][col_idx]}<br>{count}<br>{share:.0%}"
            annotations.append(
                dict(
                    x=col_label,
                    y=row_label,
                    text=text,
                    showarrow=False,
                    font=dict(color="black", size=16),
                )
            )
    return annotations


def _style_confusion_figure(fig, row_labels, col_labels):
    fig.update_layout(
        width=760,
        height=620,
        margin=dict(l=170, r=60, t=90, b=90),
    )
    fig.update_xaxes(
        side="top",
        tickmode="array",
        tickvals=col_labels,
        ticktext=col_labels,
        automargin=True,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=row_labels,
        ticktext=row_labels,
        automargin=True,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def _precision_recall(y_true, y_pred):
    tp = sum(bool(t) and bool(p) for t, p in zip(y_true, y_pred))
    fp = sum((not bool(t)) and bool(p) for t, p in zip(y_true, y_pred))
    fn = sum(bool(t) and (not bool(p)) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _oracle_auto_counts_and_metrics(df):
    y_true = df["oracle_used_web"].tolist()
    y_pred = df["auto_used_web"].tolist()
    counts_bool = _confusion_counts(y_true, y_pred)
    counts = [
        [counts_bool[1][1], counts_bool[1][0]],
        [counts_bool[0][1], counts_bool[0][0]],
    ]
    precision, recall = _precision_recall(y_true, y_pred)
    return counts, precision, recall


def _build_oracle_auto_confusion_figure(df, title=None):
    cost_rows = ["Oracle: Web Needed", "Oracle: No Web Needed"]
    cost_cols = ["Auto: Web Call", "Auto: No Web Call"]
    cost_text = [
        ["Cost Paid for Quality", "Missed Quality Gain"],
        ["Extra Cost", "Saved Cost"],
    ]
    cost_counts, precision, recall = _oracle_auto_counts_and_metrics(df)

    fig = go.Figure(
        data=go.Heatmap(
            z=cost_counts,
            x=cost_cols,
            y=cost_rows,
            xgap=2,
            ygap=2,
            colorscale="peach",
            showscale=False,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Auto decision",
        yaxis_title="Oracle decision",
        annotations=_confusion_annotations(
            cost_counts, len(df), cost_rows, cost_cols, cell_text=cost_text
        ),
    )
    fig.add_annotation(
        x=0.5,
        y=-0.18,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=f"Precision: {precision:.2%} | Recall: {recall:.2%}",
        font=dict(size=15),
    )
    fig = _style_confusion_figure(fig, cost_rows, cost_cols)
    return fig, precision, recall

def web_call_trend_over_time(df):
    df = df.copy()
    tools_categorized = load_json(f"{OUTPUT_PATH}/{PLATFORM}/metadata/all_tools_categorized.json")
    tool_to_category = {tool: cat for cat, cat_tools in tools_categorized.items() for tool, id in cat_tools.items() }
    df["categories"] = df["tools"].apply(lambda x: [tool_to_category.get(t, "Others") for t in x])

    df["month"] = pd.to_datetime(df["month"])
    df["tool_used"] = df["tools"].apply(lambda x: isinstance(x, list) and len(x) > 0)

    monthly_tooly_turns = (
        df.groupby("month")["tool_used"]
        .mean()
        .reset_index(name="tooly_turns")
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_tooly_turns["month"],
        y=monthly_tooly_turns["tooly_turns"],
        mode="lines+markers",
        name="All tool calls",
    ))

    cat = "Web & Browsing"
    df["cat_used"] = df["categories"].apply(lambda x: isinstance(x, list) and cat in x)
    monthly_tooly_turns = (
        df.groupby("month")["cat_used"]
        .mean()
        .reset_index(name="cat_tooly_turns")
        .sort_values("month")
    )
    fig.add_trace(go.Scatter(
        x=monthly_tooly_turns["month"],
        y=monthly_tooly_turns["cat_tooly_turns"],
        mode="lines+markers",
        name=f"With web call",
    ))

    df["cat_used"] = df["categories"].apply(lambda x: isinstance(x, list) and len(x) > 0 and cat not in x)
    monthly_tooly_turns = (
        df.groupby("month")["cat_used"]
        .mean()
        .reset_index(name="cat_tooly_turns")
        .sort_values("month")
    )
    fig.add_trace(go.Scatter(
        x=monthly_tooly_turns["month"],
        y=monthly_tooly_turns["cat_tooly_turns"],
        mode="lines+markers",
        name=f"Without web call",
    ))

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
    file_name = "tooly_turns_rate_over_time"
    fig.write_html(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 17))
    fig.write_image(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.pdf", format="pdf")


PLATFORM_MODELS = {
    "chatgpt": [
        'gpt-4-1', 'gpt-4-1-mini', 'gpt-4o', 'gpt-4o-mini',
        'gpt-5', 'gpt-5-instant', 'gpt-5-mini', 'gpt-5-thinking',
        'gpt-5-2', 'gpt-5-2-thinking', 'o3', 'o3-mini',
        'text-davinci-002-render-sha',
    ],
    "grok": [
        'grok-3', 'grok-3-mini-companion', 'grok-3-reasoning',
        'grok-4', 'grok-4-1-non-thinking-companion',
        'grok-4-1-non-thinking-w-tool', 'grok-4-1-thinking-1108b',
        'grok-4-1-thinking-1129', 'grok-4-auto',
        'grok-4-mini-thinking-tahoe', 'grok-420', 'imagine-image-edit',
    ],
    "claude": [],  # Claude export has no per-message model
    "deepseek": ['deepseek-chat', 'deepseek-reasoner'],
}


def web_call_trend_over_time_by_model(df):
    df = df.copy()
    selected_models = PLATFORM_MODELS.get(PLATFORM, [])
    tools_categorized = load_json(f"{OUTPUT_PATH}/{PLATFORM}/metadata/all_tools_categorized.json")
    tool_to_category = {
        tool: cat for cat, cat_tools in tools_categorized.items() for tool, id in cat_tools.items()
    }
    df["categories"] = df["tools"].apply(lambda x: [tool_to_category.get(t, "Others") for t in x])
    df["month"] = pd.to_datetime(df["month"])
    df["model"] = df["models"].apply(_primary_model)
    df = df[df["month"] >= pd.Timestamp("2024-01-01")].copy()

    cat = "Web & Browsing"
    df["cat_used"] = df["categories"].apply(lambda x: isinstance(x, list) and cat in x)
    plot_df = (
        df.groupby(["month", "model"])["cat_used"]
        .mean()
        .reset_index(name="web_call_rate")
        .sort_values(["model", "month"])
    )
    plot_df = plot_df[plot_df["model"].str.lower() != "unknown"].copy()
    plot_df = plot_df[plot_df["model"].isin(selected_models)].copy()

    color_pool = (
        qualitative.Light24
        + qualitative.Set3
        + qualitative.Alphabet
        + qualitative.Dark24
    )

    fig = go.Figure()
    for idx, model in enumerate(selected_models):
        if model not in set(plot_df["model"].unique()):
            continue
        model_df = plot_df[plot_df["model"] == model]
        fig.add_trace(
            go.Scatter(
                x=model_df["month"],
                y=model_df["web_call_rate"],
                mode="lines+markers",
                name=model,
                line=dict(color=color_pool[idx % len(color_pool)]),
                marker=dict(color=color_pool[idx % len(color_pool)]),
            )
        )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Turns with Web Call (%)",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        # margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "web_call_trend_over_time_by_model"
    fig.write_html(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 14), legend_pos=(0.8, 1.8))
    fig.write_image(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.pdf", format="pdf")


def print_available_models(df):
    df = df.copy()
    df["model"] = df["models"].apply(_primary_model)
    models = sorted(
        model
        for model in df["model"].dropna().unique()
        if isinstance(model, str) and model.lower() != "unknown"
    )
    print(models)
    return models


def topic_distribution_of_web_data(web_df):
    # in the turns that trigger web call, what is the percentage of each one: what are the important topics that had called web?
    # bar plot
    topic_counts = (
        web_df["topic"]
        .fillna("Other")
        .loc[lambda x: x != "Other"]
        .value_counts(normalize=True)
        .rename_axis("topic")
        .reset_index(name="rate")
        .sort_values("rate", ascending=False)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=topic_counts["topic"],
            y=topic_counts["rate"],
            showlegend=False
        )
    )
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Share of web-call turns",
        xaxis=dict(
            tickangle=-45,
        ),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "topic_distribution_of_web_data"
    fig.write_html(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 10))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.pdf", format="pdf")

#      topic  rate                                                                                  
# 0                           Finance  6440                                                                                  
# 1                             Games  3919                                                                                  
# 2                            Health  3442                                                                                  
# 3                   Troubleshooting  2347                                                                                  
# 4                            Travel  2284                                                                                  
# 5                Politics & History  1502                                                                                  
# 6                              Cars  1446                                                                                  
# 7                          Roleplay  1427                                                                                  
# 8                     Mental Health  1278                                                                                  
# 9                           Fashion  1268                                                                                  
# 10                    Mobile phones  1158                                                                                  
# 11                            Music  1047                                                                                  
# 12                          Cooking   992                                                                                  
# 13                   Art Generation   898                                                                                  
# 14  Social Media Content Generation   877                                                                                  
# 15                        Languages   799                                                                                  
# 16                       Job Search   710
# 17                            Books   668
# 18                  Gift Suggestion   652
# 19                              Law   635
# 20                   Email Drafting   553
# 21         Animals/Pets information   529
# 22              Weather and Climate   465
# 23                   Household Work   448
# 24                      Programming   421
# 25                           Energy   413
# 26                          Science   385
# 27                         Religion   348
# 28                        Astrology   260
# 29                           Drinks   260
# 30                             Misc   244
# 31                             Math   237
# 32               Security & Privacy   216
# 33                              GPT   191
# 34             Gender and Diversity   190
# 35                         Military   143
# 36                  Time conversion   112
# 37                              Art    76
# 38               Scientific Writing    49
# 39                   Event Planning    30

 
def topic_distriction_of_whole_data(df):
    # in the whole turns, rate of each topic calling the web over number of all turns with that topic: what topics are more prune to call web?
    # bar plot
    df["topic"] = df["topic"].fillna("Other")
    df = df[df["topic"] != "Other"].copy()

    # Platform-specific web-call detection (mirrors data_extraction.main()).
    if PLATFORM == "chatgpt":
        df["has_web_call"] = df["interactions"].apply(lambda x: "web" in str(x))
    elif PLATFORM == "claude":
        df["has_web_call"] = df["tools"].apply(
            lambda ts: any("web" in str(t).lower() for t in (ts or []))
        )
    elif PLATFORM == "grok":
        GROK_WEB_TOOLS = {
            "WebSearch", "BrowsePage", "XThreadFetch", "XSearch",
            "XUserSearch", "ViewXVideo", "ImageSearch", "PdfSearch", "PdfBrowse",
        }
        df["has_web_call"] = df["tools"].apply(
            lambda ts: any(t in GROK_WEB_TOOLS for t in (ts or []))
        )
    elif PLATFORM == "deepseek":
        DEEPSEEK_WEB_TOOLS = {
            "SEARCH", "READ_LINK", "TOOL_SEARCH", "TOOL_OPEN", "TOOL_FIND",
        }
        df["has_web_call"] = df["tools"].apply(
            lambda ts: any(t in DEEPSEEK_WEB_TOOLS for t in (ts or []))
        )
    else:
        df["has_web_call"] = False

    topic_rates = (
        df.groupby("topic")["has_web_call"]
        .mean()
        .reset_index(name="web_call_rate")
        .sort_values("web_call_rate", ascending=False)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=topic_rates["topic"],
            y=topic_rates["web_call_rate"],
            showlegend=False
        )
    )
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Web-call rate over all turns",
        xaxis=dict(
            tickangle=-45,
        ),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "topic_distribution_of_whole_data"
    fig.write_html(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 10))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.pdf", format="pdf")


def policy_distribution():
    df = pd.read_csv(f"{OUTPUT_PATH}/{PLATFORM}/metadata/web_calls_characterization.csv")
    web_df = load_web_data_from_file(fmt="pkl", platform=PLATFORM)
    web_lookup = {
        (web_row["conv_id"], web_row["turn_id"]): web_row
        for _, web_row in web_df.iterrows()
    }
    policy_label_map = {
        "High-Investment Recommendation": "High-Investment",
        "Volatile/Temporal Information": "Temporal Information",
        "Low Confidence/Niche Fact": "Low Confidence Fact",
        "Unfamiliar Term/Typo": "Unfamiliar Term",
        "High-Stakes Accuracy": "High-Stakes Accuracy",
        "External Reference": "External Reference",
        "User Verification": "User Verification",
        "Attribution/Sourcing Needed": "Attribution Needed",
        "Explicit Command": "Explicit Command",
    }
    def _safe_parse_policy(text):
        # Tolerate empty cells (judge failed) and markdown-wrapped JSON
        # (e.g. ```json ... ``` from Claude). Returns None when unparseable.
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    primary_trigger = []
    secondary_triggers = []
    policy_samples = {}
    skipped = 0
    for i, row in df.iterrows():
        policy = _safe_parse_policy(row["followed_web_policy"])
        if not isinstance(policy, dict) or "primary_trigger" not in policy:
            skipped += 1
            continue
        if policy["primary_trigger"] not in ["None of the Above", "OpenAI Product Info"]:
            primary_policy = policy["primary_trigger"]
            primary_trigger.append(primary_policy)
            if primary_policy not in policy_samples:
                web_row = web_lookup.get((row.get("conv_id"), row.get("turn_id")))
                if web_row is None or web_row.get("language") != "en":
                    continue
                user_msg_history = (
                    web_row.get("user_msg_history", [])
                )
                policy_samples[primary_policy] = {
                    "policy": primary_policy,
                    "policy_display": policy_label_map.get(primary_policy, primary_policy),
                    "prompt": str(user_msg_history[-1]).strip() if user_msg_history else "",
                    "conv_id": row.get("conv_id"),
                    "turn_id": row.get("turn_id"),
                    "topic": row.get("topic"),
                }
        secondary_triggers += [
            trigger
            for trigger in policy.get("secondary_triggers", ["None of the Above"])
            if trigger not in ["None of the Above", "OpenAI Product Info"]
        ]

    primary_rates = (
        pd.Series(primary_trigger)
        .value_counts(normalize=True)
        .rename_axis("policy")
        .reset_index(name="percentage")
    )
    primary_rates["policy_display"] = primary_rates["policy"].apply(
        lambda x: policy_label_map.get(x, x)
    )
    secondary_rates = (
        pd.Series(secondary_triggers)
        .value_counts(normalize=True)
        .rename_axis("policy")
        .reset_index(name="percentage")
    )
    secondary_rates["policy_display"] = secondary_rates["policy"].apply(
        lambda x: policy_label_map.get(x, x)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=primary_rates["policy_display"],
            y=primary_rates["percentage"],
            text=primary_rates["percentage"].apply(lambda x: f"{x:.1%}"),
            # name="Primary trigger",
            showlegend=False
        )
    )
    # fig.add_trace(
    #     go.Bar(
    #         x=secondary_rates["policy"],
    #         y=secondary_rates["percentage"],
    #         name="Secondary trigger",
    #     )
    # )
    fig.update_layout(
        xaxis_title="Trigger",
        yaxis_title="Percentage",
        barmode="group",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "web_call_policy_characterization"
    fig.write_html(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18))
    fig.update_xaxes(tickangle=-45)
    fig.update_xaxes(tickfont=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=16))
    fig.write_image(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.pdf", format="pdf")
    to_json(
        [
            policy_samples[policy]
            for policy in primary_rates["policy"].tolist()
            if policy in policy_samples
        ],
        f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/web_call_policy_characterization_samples.json",
    )


def policy_distribution_stacked_by_topics():
    df = pd.read_csv(f"{OUTPUT_PATH}/{PLATFORM}/metadata/web_calls_characterization.csv")
    # selected_topics = ['Health', 'Travel', 'Finance', 'Politics & History', 'Science']
    selected_topics = ['Travel', 'Cars', 'Mobile phones', 'Finance', 'Games', 'Health']
    policy_label_map = {
        "High-Investment Recommendation": "High-Investment",
        "Volatile/Temporal Information": "Temporal Information",
        "Low Confidence/Niche Fact": "Low Confidence Fact",
        "Unfamiliar Term/Typo": "Unfamiliar Term",
        "High-Stakes Accuracy": "High-Stakes Accuracy",
        "External Reference": "External Reference",
        "User Verification": "User Verification",
        "Attribution/Sourcing Needed": "Attribution Needed",
        "Explicit Command": "Explicit Command",
    }

    def _safe_parse_policy(text):
        # Tolerate empty cells and markdown-wrapped JSON (```json ... ```).
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    plot_rows = []
    for _, row in df.iterrows():
        if row["topic"] not in selected_topics:
            continue
        policy = _safe_parse_policy(row["followed_web_policy"])
        if not isinstance(policy, dict) or "primary_trigger" not in policy:
            continue
        if policy["primary_trigger"] in ["None of the Above", "OpenAI Product Info"]:
            continue
        plot_rows.append(
            {
                "topic": row["topic"],
                "primary_trigger": policy["primary_trigger"],
            }
        )

    plot_df = pd.DataFrame(plot_rows)
    if len(plot_df) == 0:
        return

    shares = (
        plot_df.groupby(["primary_trigger", "topic"])
        .size()
        .reset_index(name="count")
    )
    shares["primary_trigger_display"] = shares["primary_trigger"].apply(
        lambda x: policy_label_map.get(x, x)
    )
    shares["percentage"] = shares.groupby("primary_trigger")["count"].transform(
        lambda x: x / x.sum()
    )
    primary_rates = (
        pd.Series(plot_df["primary_trigger"])
        .value_counts(normalize=True)
        .rename_axis("policy")
        .reset_index(name="percentage")
    )
    primary_rates["policy_display"] = primary_rates["policy"].apply(
        lambda x: policy_label_map.get(x, x)
    )
    policy_order = primary_rates["policy_display"].tolist()

    fig = go.Figure()
    for topic in selected_topics:
        topic_counts = shares[shares["topic"] == topic].copy()
        topic_counts["primary_trigger_display"] = pd.Categorical(
            topic_counts["primary_trigger_display"],
            categories=policy_order,
            ordered=True,
        )
        topic_counts = topic_counts.sort_values("primary_trigger_display")
        fig.add_trace(
            go.Bar(
                x=topic_counts["primary_trigger_display"],
                y=topic_counts["percentage"],
                name=topic,
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Trigger",
        yaxis_title="Percentage",
        xaxis=dict(
            categoryorder="array",
            categoryarray=policy_order,
        ),
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    file_name = "web_call_policy_by_topic"
    fig.write_html(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16), legend_pos=(0.8, 1.4))
    fig.update_xaxes(tickangle=-45)
    fig.update_xaxes(tickfont=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=16))
    fig.write_image(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{file_name}.pdf", format="pdf")


def replay_evaluations(
    eval_model_name="gpt-5.4-nano-2026-03-17",
    llm_model_name="gpt-5.4-nano-2026-03-17",
    temperature="0.0",
    web_tool_type="openai"
):
    if web_tool_type == "openai":
        base_dir = f"{OUTPUT_PATH}/{PLATFORM}/metadata/preference_evaluation/{eval_model_name}/{temperature}"
    else:
        base_dir = f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}"

    df = pd.read_csv(f"{base_dir}/{llm_model_name}.csv").copy()

    if len(df) == 0:
        return
    response_modes = ["auto", "none", "required"]
    if llm_model_name == "invivo":
        response_modes.append("invivo")

    required_output_cols = [f"{mode}_output_text" for mode in ["none", "required", "auto"]]
    original_len = len(df)
    df = df[
        df.apply(
            lambda row: all(
                str(row.get(col, "")).strip() != ""
                for col in required_output_cols
            ),
            axis=1,
        )
    ].copy()
    dropped_rows = original_len - len(df)
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows without required replay outputs.")
    if len(df) == 0:
        return

    likert_metrics = [
        "factuality_5likert",
        "relevance_5likert",
        # "informativeness_5likert",
        "completeness_5likert",
    ]
    metric_label_map = {
        "factuality_5likert": "Factuality",
        # "informativeness_5likert": "Informativeness",
        "completeness_5likert": "Completeness",
        "relevance_5likert": "Relevance",
    }
    mode_label_map = {
        "auto": "Auto",
        "none": "None",
        "required": "Required",
        "invivo": "INVIVO",
    }

    def _to_numeric(series):
        return pd.to_numeric(series, errors="coerce")

    metric_labels = [
        metric_label_map.get(metric, metric.replace("_", " ").title())
        for metric in likert_metrics
    ]
    summary_rows = []
    for mode in response_modes:
        row = {"Response Mode": mode_label_map.get(mode, mode.title())}
        metric_values = []
        for metric, metric_label in zip(likert_metrics, metric_labels):
            score_col = f"{mode}_{metric}_score"
            score = (
                _to_numeric(df[score_col]).mean()
                if score_col in df.columns
                else float("nan")
            )
            row[metric_label] = score
            metric_values.append(score)
        row["Average"] = pd.Series(metric_values, dtype="float64").mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    display_df = summary_df.copy()
    for col in metric_labels + ["Average"]:
        display_df[col] = display_df[col].apply(
            lambda value: "NA" if pd.isna(value) else f"{value:.2f}"
        )

    print(web_tool_type)
    print(
        "\nReplay evaluation average 5-Likert scores "
        f"({llm_model_name}, judged by {eval_model_name}):"
    )
    print(display_df.to_string(index=False))
    return summary_df


def replay_call_outcome_summary(
    eval_model_name="gpt-5.4-nano-2026-03-17",
    llm_model_name="gpt-5.4-nano-2026-03-17",
    temperature="0.0",
    web_tool_type="openai"
):
    if web_tool_type == "openai":
        base_dir = f"{OUTPUT_PATH}/{PLATFORM}/metadata/preference_evaluation/{eval_model_name}/{temperature}"
    else:
        base_dir = f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}"

    df = pd.read_csv(f"{base_dir}/{llm_model_name}.csv").copy()

    if len(df) == 0:
        return

    required_output_cols = [f"{mode}_output_text" for mode in ["none", "required", "auto"]]
    df = df[
        df.apply(
            lambda row: all(
                str(row.get(col, "")).strip() != ""
                for col in required_output_cols
            ),
            axis=1,
        )
    ].copy()
    if len(df) == 0:
        return

    likert_metrics = [
        "factuality_5likert",
        "relevance_5likert",
        # "informativeness_5likert",
        "completeness_5likert",
    ]

    def _web_call_flag_series(data_frame, cols):
        flags = pd.Series(False, index=data_frame.index)
        for col in cols:
            if col in data_frame.columns:
                flags = flags | data_frame[col].apply(_parse_bool).astype(bool)
        return flags

    if llm_model_name == "invivo":
        df["decision_uses_web"] = _web_call_flag_series(
            df, ["invivo_called_web"]
        )
        decision_label = "InVivo"
        decision_file_suffix = "invivo"
    else:
        df["decision_uses_web"] = _web_call_flag_series(
            df, ["auto_called_web"]
        )
        decision_label = "Auto"
        decision_file_suffix = "auto"

    def _call_outcome(row):
        if row["best_call"] == "tie":
            return "Right Call"
        if row["decision_uses_web"] and row["best_call"] == "required":
            return "Right Call"
        if (not row["decision_uses_web"]) and row["best_call"] == "none":
            return "Right Call"
        if row["decision_uses_web"] and row["best_call"] == "none":
            return "Over Call"
        if (not row["decision_uses_web"]) and row["best_call"] == "required":
            return "Under Call"
        return "Right Call"

    def _plot_outcome_for_metrics(metric_names, suffix, title):
        plot_df = df.copy()
        score_cols_web = [f"required_{metric}_score" for metric in metric_names]
        score_cols_nonweb = [f"none_{metric}_score" for metric in metric_names]
        plot_df["required_avg_score"] = plot_df[score_cols_web].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        plot_df["none_avg_score"] = plot_df[score_cols_nonweb].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        plot_df["best_call"] = plot_df.apply(
            lambda row: (
                "required"
                if row["required_avg_score"] > row["none_avg_score"]
                else (
                    "none"
                    if row["required_avg_score"] < row["none_avg_score"]
                    else "tie"
                )
            ),
            axis=1,
        )
        plot_df["call_outcome"] = plot_df.apply(_call_outcome, axis=1)
        outcome_order = ["Right Call", "Over Call", "Under Call"]
        stacked_counts = (
            plot_df.groupby(["call_outcome", "decision_uses_web"])
            .size()
            .unstack(fill_value=0)
            .reindex(outcome_order, fill_value=0)
        )
        stacked_counts = stacked_counts.reindex(
            columns=[False, True], fill_value=0
        ).astype(int)
        no_web_counts = stacked_counts[False].tolist()
        web_counts = stacked_counts[True].tolist()
        total_counts = stacked_counts.sum(axis=1).astype(int).tolist()
        text_no_web = [str(count) if count else "" for count in no_web_counts]
        text_web = [str(count) if count else "" for count in web_counts]
        text_total = [str(count) if count else "" for count in total_counts]

        outcome_fig = go.Figure()
        outcome_fig.add_trace(
            go.Bar(
                x=outcome_order,
                y=no_web_counts,
                name="No Web Call",
                text=text_no_web,
                textposition="inside",
                marker_color=["#54A24B", "#E45756", "#F58518"],
                hovertemplate="%{x}<br>No web call: %{y}<extra></extra>",
            )
        )
        outcome_fig.add_trace(
            go.Bar(
                x=outcome_order,
                y=web_counts,
                name="With Web Call",
                text=text_web,
                textposition="inside",
                marker=dict(
                    color=["#54A24B", "#E45756", "#F58518"],
                    pattern=dict(
                        shape="/",
                        fgcolor="#000000",
                        size=8,
                        solidity=0.25,
                    ),
                ),
                hovertemplate="%{x}<br>With web call: %{y}<extra></extra>",
            )
        )
        outcome_fig.add_trace(
            go.Scatter(
                x=outcome_order,
                y=total_counts,
                mode="text",
                text=text_total,
                textposition="top center",
                textfont=dict(color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        outcome_fig.update_layout(
            title=title,
            xaxis_title=f"{decision_label} Call Type",
            yaxis_title="Samples",
            barmode="stack",
            uniformtext_minsize=12,
            uniformtext_mode="hide",
        )
        outcome_fig.update_yaxes(
            range=[0, max(max(total_counts) * 1.2, 1)]
        )
        file_name = (
            f"replay_{llm_model_name}_evaluations_{eval_model_name}_"
            f"{decision_file_suffix}_call_outcome_{suffix}"
        )
        os.makedirs(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/replay_modes/{web_tool_type}/", exist_ok=True)
        # outcome_fig.write_html(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/{web_tool_type}/{file_name}.html")
        outcome_fig = with_paper_style(
            outcome_fig, config=styler(18, 18), legend_pos=(0.5, 1.15)
        )
        outcome_fig.update_layout(
            legend=dict(
                x=0.5,
                y=1.15,
                xanchor="center",
                yanchor="top",
                orientation="h",
            )
        )
        outcome_fig.write_image(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/replay_modes/{web_tool_type}/{file_name}.pdf", format="pdf")

    # _plot_outcome_for_metrics(binary_metrics, "binary", "Binary Call Outcome")
    _plot_outcome_for_metrics(likert_metrics, "5likert", "5-Likert Call Outcome")


def replay_call_outcome_venn_diagram(
    eval_model_name="gpt-5.4-nano-2026-03-17",
    llm_model_name="gpt-5.4-nano-2026-03-17",
    temperature="0.0",
):
    tool_configs = [
        {
            "tool": "openai",
            "label": "OpenAI",
            "base_dir": f"{OUTPUT_PATH}/{PLATFORM}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": llm_model_name,
        },
        {
            "tool": "tavily",
            "label": "Tavily",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": f"{llm_model_name}_responses_url_mcp-tavily",
        },
        {
            "tool": "serpapi",
            "label": "SerpAPI",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": f"{llm_model_name}_mcp-serp",
        },
        {
            "tool": "perplexity",
            "label": "Perplexity",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": f"{llm_model_name}_responses_url_mcp-perplexity",
        },
        {
            "tool": "brave",
            "label": "Brave",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": f"{llm_model_name}_responses_url_mcp-brave",
        },
    ]
    tool_order = [config["tool"] for config in tool_configs]
    tool_label_map = {
        config["tool"]: config["label"]
        for config in tool_configs
    }
    merge_keys = ["prompt", "sample_source", "conv_id", "turn_id"]

    def _web_call_flag_series(data_frame, cols):
        flags = pd.Series(False, index=data_frame.index)
        for col in cols:
            if col in data_frame.columns:
                flags = flags | data_frame[col].apply(_parse_bool).astype(bool)
        return flags

    def _load_tool_df(config):
        file_path = f"{config['base_dir']}/{config['file_name']}.csv"
        if not os.path.exists(file_path):
            print(f"Missing evaluation file for {config['label']}: {file_path}")
            return pd.DataFrame()

        df = pd.read_csv(file_path).copy()
        if len(df) == 0:
            return df
        missing_keys = [col for col in merge_keys if col not in df.columns]
        if missing_keys:
            print(
                f"Missing sample key columns for {config['label']}: "
                f"{', '.join(missing_keys)}"
            )
            return pd.DataFrame()
        if "auto_output_text" in df.columns:
            df = df[
                df["auto_output_text"].apply(
                    lambda value: str(value).strip() != ""
                )
            ].copy()
        df = df[
            df.apply(
                lambda row: all(
                    str(row.get(col, "")).strip() != ""
                    for col in merge_keys
                ),
                axis=1,
            )
        ].copy()
        df = df.drop_duplicates(subset=merge_keys).copy()
        df[config["tool"]] = _web_call_flag_series(
            df, ["auto_called_web", "auto_web_call"]
        )
        return df[merge_keys + [config["tool"]]].copy()

    tool_dfs = []
    for config in tool_configs:
        tool_df = _load_tool_df(config)
        if len(tool_df) == 0:
            print(f"No rows available for {config['label']}.")
            return pd.DataFrame()
        tool_dfs.append(tool_df)

    merged_df = tool_dfs[0]
    for tool_df in tool_dfs[1:]:
        merged_df = merged_df.merge(tool_df, on=merge_keys, how="inner")
    if len(merged_df) == 0:
        print("No common samples found across all five tools.")
        return pd.DataFrame()

    combo_counts = Counter(
        tuple(tool for tool in tool_order if row[tool])
        for _, row in merged_df.iterrows()
    )
    all_tools_combo = tuple(tool_order)
    no_tool_count = combo_counts.get(tuple(), 0)
    all_tools_count = combo_counts.get(all_tools_combo, 0)
    intersection_rows = [
        {
            "tools": combo,
            "tool_count": len(combo),
            "count": count,
            "tools_display": ", ".join(tool_label_map[tool] for tool in combo),
        }
        for combo, count in combo_counts.items()
        if combo
    ]
    intersection_rows = sorted(
        intersection_rows,
        key=lambda row: (
            -row["tool_count"],
            -row["count"],
            row["tools_display"],
        ),
    )
    summary_rows = [
        {
            "tools": row["tools_display"],
            "tool_count": row["tool_count"],
            "count": row["count"],
        }
        for row in intersection_rows
    ]
    summary_rows.append(
        {
            "tools": "No Tool",
            "tool_count": 0,
            "count": no_tool_count,
        }
    )
    summary_df = pd.DataFrame(summary_rows)

    print(f"Common samples across all five tools: {len(merged_df)}")
    print(f"Samples where all five tools called web in auto mode: {all_tools_count}")
    print(f"Samples where no tool called web in auto mode: {no_tool_count}")
    print("\nAuto web-call exact intersections:")
    print(summary_df.to_string(index=False))

    output_dir = f"{OUTPUT_PATH}/{PLATFORM}/{CONF}/replay_modes/"
    os.makedirs(output_dir, exist_ok=True)
    file_name = (
        f"replay_{llm_model_name}_auto_web_call_venn_"
        f"{eval_model_name}"
    )
    summary_df.to_csv(f"{output_dir}/{file_name}_summary.csv", index=False)
    to_json(
        summary_df.to_dict(orient="records"),
        f"{output_dir}/{file_name}_summary.json",
    )

    matrix_rows = intersection_rows + [
        {
            "tools": tuple(),
            "tool_count": 0,
            "count": no_tool_count,
            "tools_display": "No Tool",
        }
    ]
    dot = "●"
    empty = ""
    tool_columns = [
        [
            dot if tool in row["tools"] else empty
            for row in matrix_rows
        ]
        for tool in tool_order
    ]
    count_column = [row["count"] for row in matrix_rows]
    row_fill = [
        "#F7F7F7" if idx % 2 == 0 else "white"
        for idx in range(len(matrix_rows))
    ]
    tool_fill_columns = [row_fill for _ in tool_order]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[1.15, 1.15, 1.15, 1.35, 1.15, 1.0],
                header=dict(
                    values=[
                        *[tool_label_map[tool] for tool in tool_order],
                        "Count",
                    ],
                    fill_color="#E8EEF6",
                    align=["center", "center", "center", "center", "center", "center"],
                    font=dict(color="black", size=16),
                    height=38,
                    line_color="white",
                ),
                cells=dict(
                    values=[
                        *tool_columns,
                        count_column,
                    ],
                    fill_color=[
                        *tool_fill_columns,
                        row_fill,
                    ],
                    align=["center", "center", "center", "center", "center", "center"],
                    font=dict(
                        color=[
                            *[["black"] * len(matrix_rows) for _ in tool_order],
                            ["black"] * len(matrix_rows),
                        ],
                        size=[
                            *[22 for _ in tool_order],
                            16,
                        ],
                    ),
                    height=42,
                    line_color="white",
                ),
            )
        ]
    )
    fig.update_layout(
        width=760,
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Open Sans",
    )
    fig.write_html(f"{output_dir}/{file_name}.html")
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

    return summary_df


def subset_selection_for_policy_evaluation_by_human():
    df = pd.read_csv(f"{OUTPUT_PATH}/{PLATFORM}/metadata/web_calls_characterization.csv").reset_index()
    subset = df.sample(100)
    subset.to_csv(f"{OUTPUT_PATH}/{PLATFORM}/metadata/web_calls_characterization_subset_for_human_eval.csv", index=False)


def count_model_used(web_df):
    # Count model usage across all web-call turns. A turn can list multiple models,
    # so we report both total mentions and the final/primary model per turn.
    df = web_df.copy()

    if "models" not in df.columns:
        print("Column `models` not found.")
        return pd.DataFrame(), pd.DataFrame()

    def _ensure_model_list(value):
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

    df["models"] = df["models"].apply(_ensure_model_list)
    df["primary_model"] = df["models"].apply(_primary_model)

    all_model_counts = (
        df["models"]
        .explode()
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="count")
    )

#     Model mentions across `models`:
#                       model  count
#                      gpt-4o 110326
#                       gpt-5  46854
#                          o3   9012
#                  gpt-5-mini   8005
#              gpt-5-thinking   4459
#                 gpt-4o-mini   3906
#                gpt-5-t-mini   3719
#                     o4-mini   3270
#                     gpt-5-2   3105
#                     o3-mini   2725
#                gpt-4-1-mini   2545
#                     gpt-5-1   2147
#            gpt-5-1-thinking   1392
#                o4-mini-high    908
#                o3-mini-high    260
#                     gpt-4-1    175
#            gpt-5-2-thinking    123
#                     gpt-4-5     98
# text-davinci-002-render-sha     44
#              gpt-5-a-t-mini     27
#         gpt-5-auto-thinking     18
#               gpt-5-instant      7
#                    research      4

    primary_model_counts = (
        df["primary_model"]
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="count")
    )

#     Primary model per turn:
#                       model  count
#                      gpt-4o  23706
#                       gpt-5   9997
#                  gpt-5-mini   2381
#                 gpt-4o-mini   1247
#                gpt-4-1-mini    809
#                     Unknown    717
#                gpt-5-t-mini    714
#                     gpt-5-2    663
#              gpt-5-thinking    475
#                     gpt-5-1    431
#                     o4-mini    326
#                          o3    268
#                     o3-mini    131
#                o4-mini-high     79
#            gpt-5-1-thinking     54
#                     gpt-4-1     38
#                     gpt-4-5     25
#                o3-mini-high     14
# text-davinci-002-render-sha     11
#            gpt-5-2-thinking      8
#              gpt-5-a-t-mini      4
#         gpt-5-auto-thinking      3
#               gpt-5-instant      2
#                    research      1

    print("\nModel mentions across `models`:")
    print(all_model_counts.to_string(index=False))

    print("\nPrimary model per turn:")
    print(primary_model_counts.to_string(index=False))

    return all_model_counts, primary_model_counts


def _clean_query_type_label(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def _distinct_query_type_count(row, cols):
    return len(
        {
            _clean_query_type_label(value)
            for value in row[cols]
            if _clean_query_type_label(value)
        }
    )


def _majority_query_type(values):
    labels = [_clean_query_type_label(value) for value in values]
    labels = [label for label in labels if label]
    if not labels:
        return ""

    counts = Counter(labels)
    top_count = max(counts.values())
    top_labels = sorted(
        label for label, count in counts.items() if count == top_count
    )
    if len(top_labels) == 1:
        return top_labels[0]
    return "tie: " + "|".join(top_labels)


def _non_tie_label(value):
    label = _clean_query_type_label(value)
    return label and not label.startswith("tie:")


def _fleiss_kappa(label_df):
    label_df = label_df.apply(lambda col: col.apply(_clean_query_type_label))
    complete_df = label_df[
        label_df.apply(lambda row: all(bool(label) for label in row), axis=1)
    ].copy()
    if len(complete_df) == 0 or complete_df.shape[1] < 2:
        return {
            "fleiss_kappa": float("nan"),
            "num_items": len(complete_df),
            "num_raters": complete_df.shape[1],
        }

    categories = sorted(
        {
            label
            for col in complete_df.columns
            for label in complete_df[col].tolist()
            if label
        }
    )
    num_items = len(complete_df)
    num_raters = complete_df.shape[1]
    if not categories or num_raters < 2:
        return {
            "fleiss_kappa": float("nan"),
            "num_items": num_items,
            "num_raters": num_raters,
        }

    count_rows = []
    for _, row in complete_df.iterrows():
        counts = Counter(row.tolist())
        count_rows.append([counts.get(category, 0) for category in categories])
    count_df = pd.DataFrame(count_rows, columns=categories)

    item_agreement = (
        (count_df.pow(2).sum(axis=1) - num_raters)
        / (num_raters * (num_raters - 1))
    )
    observed_agreement = item_agreement.mean()
    category_proportions = count_df.sum(axis=0) / (num_items * num_raters)
    expected_agreement = category_proportions.pow(2).sum()
    denominator = 1 - expected_agreement
    fleiss_kappa = (
        (observed_agreement - expected_agreement) / denominator
        if denominator
        else float("nan")
    )

    return {
        "fleiss_kappa": fleiss_kappa,
        "num_items": num_items,
        "num_raters": num_raters,
        "observed_agreement": observed_agreement,
        "expected_agreement": expected_agreement,
        "categories": "|".join(categories),
    }

def df_creaton_for_annotation():
    # Create one comparison table for query-type labels assigned by three judges.
    # We align on the replay prompts and keep the prompt plus the label/reasoning
    # from each judge so the rows are ready for annotation.
    temperature = "0.0"
    replay_model_name = "gpt-4o-mini"
    judge_models = [
        "gpt-4o-mini",
        "gpt-5.4-mini",
        "gpt-5.4-nano-2026-03-17",
    ]

    merge_keys = ["prompt", "sample_source", "conv_id", "turn_id"]
    merged_df = None
    annotation_label_options = [
        "informational",
        "exploratory",
        "transactional",
        "navigational",
    ]

    for judge_model in judge_models:
        path = (
            f"{OUTPUT_PATH}/{PLATFORM}/metadata/preference_evaluation/"
            f"{judge_model}/{temperature}/{replay_model_name}.csv"
        )
        if not os.path.exists(path):
            print(f"Missing file for {judge_model}: {path}")
            continue

        judge_df = pd.read_csv(path)
        keep_cols = [
            col
            for col in merge_keys + ["query_type", "query_type_reasoning"]
            if col in judge_df.columns
        ]
        judge_df = judge_df[keep_cols].copy()
        judge_df = judge_df.rename(
            columns={
                "query_type": f"query_type_judge_{judge_model}",
                "query_type_reasoning": f"query_type_judge_{judge_model}_reasoning",
            }
        )

        if merged_df is None:
            merged_df = judge_df
        else:
            merged_df = merged_df.merge(judge_df, on=merge_keys, how="outer")

    if merged_df is None or len(merged_df) == 0:
        print(
            "No preference-evaluation files found to build the annotation dataframe."
        )
        return pd.DataFrame()

    query_type_cols = [
        f"query_type_judge_{judge_model}"
        for judge_model in judge_models
    ]
    present_query_type_cols = [
        col for col in query_type_cols if col in merged_df.columns
    ]
    merged_df["num_distinct_query_types"] = merged_df[present_query_type_cols].apply(
        lambda row: len(
            {
                str(value).strip()
                for value in row
                if pd.notna(value) and str(value).strip()
            }
        ),
        axis=1,
    )
    merged_df["all_judges_agree"] = merged_df["num_distinct_query_types"] <= 1
    full_merged_df = merged_df.copy()
    merged_df["annotator_query_type"] = ""
    merged_df["annotator_reasoning"] = ""
    merged_df["annotator_label_options"] = ", ".join(annotation_label_options)

    ordered_cols = [col for col in merge_keys if col != "sample_source"]
    # ordered_cols += present_query_type_cols
    ordered_cols += [
        "annotator_query_type",
        "annotator_reasoning",
        "annotator_label_options",
        # "num_distinct_query_types",
        # "all_judges_agree",
    ]
    merged_df = merged_df[ordered_cols].sort_values(
        [
            # "all_judges_agree", 
            # "num_distinct_query_types", 
            "prompt"
        ],
        ascending=[
            # True, 
            # False, 
            True
        ],
    ).reset_index(drop=True)

    output_base = f"{OUTPUT_PATH}/{PLATFORM}/metadata/query_type_annotation_comparison"
    merged_df.to_csv(f"{output_base}.csv", index=False)
    merged_df.to_pickle(f"{output_base}.pkl")
    to_json(merged_df.to_dict(orient="records"), f"{output_base}.json")

    print(f"Saved {len(merged_df)} rows to {output_base}.csv/.pkl/.json")
    query_type_annotation_comparison(
        judge_df=full_merged_df,
        judge_models=judge_models,
        output_base=output_base,
    )
    return merged_df


if __name__ == "__main__":
    # args = parse_args()
    PLATFORM = "args.platform"
    # OUTPUT_PATH = "./outputs"
    # os.makedirs(f"{OUTPUT_PATH}/{PLATFORM}/{CONF}", exist_ok=True)

    full_df = load_whole_data_from_file(fmt="pkl", platform=PLATFORM)
    # Drop rows with bad/missing timestamps (e.g., epoch 0 → 1970-01-01)
    # bad = full_df["time"] < pd.Timestamp("2020-01-01")
    # if bad.any():
    #     print(f"Dropping {int(bad.sum())} rows with timestamp before 2020 from full_df.")
    #     full_df = full_df[~bad].copy()
    # # print("# all turns:", len(full_df))
    # web_df = load_web_data_from_file(fmt="pkl", platform=PLATFORM)
    # bad = web_df["time"] < pd.Timestamp("2020-01-01")
    # if bad.any():
    #     print(f"Dropping {int(bad.sum())} rows with timestamp before 2020 from web_df.")
    #     web_df = web_df[~bad].copy()
    # print("# turns with web call:", len(web_df))

    web_call_trend_over_time(full_df)
    # print_available_models(full_df)
    # web_call_trend_over_time_by_model(full_df)
    # topic_distriction_of_whole_data(full_df)
    # topic_distribution_of_web_data(web_df)
    # policy_distribution()
    # policy_distribution_stacked_by_topics()
    # replay_evaluations(llm_model_name="invivo")
    # subset_selection_for_policy_evaluation_by_human()
    # replay_call_outcome_summary(llm_model_name="invivo")

    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07", eval_model_name="gpt-5.4-mini", web_tool_type="openai")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07", eval_model_name="gpt-5.4-mini", web_tool_type="openai")
    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07_mcp-serp", eval_model_name="gpt-5.4-mini", web_tool_type="serp")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07_mcp-serp", eval_model_name="gpt-5.4-mini", web_tool_type="serp")
    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-brave", eval_model_name="gpt-5.4-mini", web_tool_type="brave")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-brave", eval_model_name="gpt-5.4-mini", web_tool_type="brave")
    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-tavily", eval_model_name="gpt-5.4-mini", web_tool_type="tavily")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-tavily", eval_model_name="gpt-5.4-mini", web_tool_type="tavily")
    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-perplexity", eval_model_name="gpt-5.4-mini", web_tool_type="perplexity")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-perplexity", eval_model_name="gpt-5.4-mini", web_tool_type="perplexity")
    
    # replay_call_outcome_venn_diagram(llm_model_name="gpt-5-mini-2025-08-07", eval_model_name="gpt-5.4-mini")

    # count_model_used(web_df)
    # df_creaton_for_annotation()
