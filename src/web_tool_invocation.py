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
from utils import *
from data_utils import *
from paper import with_paper_style, styler
from data_extraction import load_web_data_from_file, load_whole_data_from_file
from data_extraction_other_cai import load_whole_data_from_file as load_whole_data_from_file_cai

CONF = "emnlp/web_tool_invocation"

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


def _is_missing_value(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_possible_literal(value):
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text[0] not in "[{":
        return value

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
    return value


def _as_list_value(value):
    if _is_missing_value(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            return _as_list_value(value.tolist())
        except Exception:
            pass
    parsed = _parse_possible_literal(value)
    if parsed is not value:
        return _as_list_value(parsed)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [value]


def _stringify_nested_text(value):
    if _is_missing_value(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        parsed = _parse_possible_literal(text)
        if parsed is not text:
            return _stringify_nested_text(parsed)
        return text
    if isinstance(value, dict):
        for key in ("content", "text", "q"):
            if key in value:
                text = _stringify_nested_text(value.get(key))
                if text:
                    return text
        return "\n".join(
            text
            for text in (_stringify_nested_text(item) for item in value.values())
            if text
        ).strip()
    if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
        return "\n".join(
            text
            for text in (_stringify_nested_text(item) for item in value)
            if text
        ).strip()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _stringify_nested_text(value.tolist())
        except Exception:
            pass
    return str(value).strip()


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


def _trim_text(text, max_chars):
    text = str(text or "").strip()
    if not max_chars or len(text) <= max_chars:
        return text
    return text[-max_chars:].strip()


def _json_safe_value(value):
    if _is_missing_value(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for key, val in value.items()}
    if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
        return [_json_safe_value(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe_value(value.tolist())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except Exception:
            pass
    return str(value)


def _metric_preference_counts(row, metric_names):
    prefer_none = 0
    prefer_required = 0
    ties = 0
    for metric in metric_names:
        none_score = pd.to_numeric(
            row.get(f"none_{metric}_score"), errors="coerce"
        )
        required_score = pd.to_numeric(
            row.get(f"required_{metric}_score"), errors="coerce"
        )
        if pd.isna(none_score) or pd.isna(required_score):
            continue
        if required_score > none_score:
            prefer_required += 1
        elif none_score > required_score:
            prefer_none += 1
        else:
            ties += 1

    return prefer_none, prefer_required, ties


def _strict_call_outcome(row, metric_names):
    prefer_none, prefer_required, _ = _metric_preference_counts(
        row, metric_names
    )
    total_metrics = len(metric_names)
    decision_uses_web = _parse_bool(row.get("decision_uses_web", False))

    # Right call if at least one selected metric agrees with the decision.
    if decision_uses_web:
        if prefer_required >= 1:
            return "Right Call"
        if prefer_none >= 1:
            return "Over Call"
        return "Right Call"

    if prefer_none >= 1:
        return "Right Call"
    if prefer_required == total_metrics and total_metrics > 0:
        return "Under Call"
    return "Right Call"


def _row_has_web_call(row):
    for col in ("tools", "interactions"):
        if col not in row:
            continue
        if any("web" in str(item).lower() for item in _as_list_value(row.get(col))):
            return True

    if "web_queries" in row:
        web_queries = _as_list_value(row.get("web_queries"))
        return bool(_stringify_nested_text(web_queries))

    return True


def _latest_user_message(row):
    user_msg_history = _as_list_value(row.get("user_msg_history", []))
    user_messages = [
        str(message).strip()
        for message in user_msg_history
        if str(message).strip()
    ]
    if user_messages:
        return user_messages[-1]

    for col in ("user_query", "prompt", "Prompt_with_history"):
        text = _stringify_nested_text(row.get(col, ""))
        if text:
            return text
    return ""


def _turn_messages(row):
    turn_msgs = row.get("turn_msgs", [])
    parsed = _parse_possible_literal(turn_msgs)
    return parsed if isinstance(parsed, list) else []


def _message_role(message):
    author = message.get("author", {}) if isinstance(message, dict) else {}
    return str(author.get("name") or author.get("role") or "").strip()


def _message_thought_text(message):
    if not isinstance(message, dict):
        return ""
    content = message.get("content", {})
    if not isinstance(content, dict):
        return ""
    if content.get("content_type") != "thoughts":
        return ""
    return _stringify_nested_text(content.get("thoughts", []))


def _message_has_web_call_or_query(message):
    if not isinstance(message, dict):
        return False

    role = _message_role(message)
    recipient = str(message.get("recipient", "")).lower()
    if role == "assistant" and recipient and recipient != "all" and "web" in recipient:
        return True

    metadata = message.get("metadata", {})
    if not isinstance(metadata, dict):
        return False

    if _as_list_value(metadata.get("search_queries", [])):
        return True

    search_model_queries = metadata.get("search_model_queries", {})
    if isinstance(search_model_queries, dict):
        if _as_list_value(search_model_queries.get("queries", [])):
            return True

    return False


def _thoughts_before_first_web_call(row):
    thoughts = []
    for message in _turn_messages(row):
        thought_text = _message_thought_text(message)
        if thought_text:
            thoughts.append(thought_text)

        if _message_has_web_call_or_query(message):
            break

    return "\n".join(thoughts).strip()


def _thoughts_before_first_web_query(row):
    web_queries = _as_list_value(row.get("web_queries", []))
    if not _stringify_nested_text(web_queries):
        return ""

    thoughts_list = _as_list_value(row.get("thoughts_list", []))
    if not thoughts_list:
        return ""

    return _stringify_nested_text(thoughts_list[0])


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
    tools_categorized = load_json(f"{OUTPUT_PATH}/metadata/all_tools_categorized.json")
    tool_to_category = {tool: cat for cat, cat_tools in tools_categorized.items() for tool, id in cat_tools.items() }
    df["categories"] = df["tools"].apply(lambda x: [tool_to_category[t] for t in x])

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
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 17))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def web_call_trend_over_time_all_convai(df):
    min_valid_month = pd.Timestamp("2023-01-01")
    cat = "Web & Browsing"

    def _monthly_web_rate(platform_df, tools_categorized):
        platform_df = platform_df.copy()
        tool_to_category = {}
        for cat_name, cat_tools in (tools_categorized or {}).items():
            if isinstance(cat_tools, dict):
                for tool_name in cat_tools:
                    tool_to_category[tool_name] = cat_name

        platform_df["tools"] = platform_df["tools"].apply(_as_list_value)
        platform_df["categories"] = platform_df["tools"].apply(
            lambda tools: [tool_to_category.get(tool, "Others") for tool in tools]
        )
        platform_df["month"] = (
            pd.to_datetime(platform_df["month"], errors="coerce", utc=True)
            .dt.tz_convert(None)
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        platform_df = platform_df.dropna(subset=["month"])
        platform_df = platform_df[platform_df["month"] >= min_valid_month].copy()

        platform_df["cat_used"] = platform_df["categories"].apply(lambda cats: cat in cats)
        return (
            platform_df.groupby("month")["cat_used"]
            .mean()
            .reset_index(name="cat_tooly_turns")
            .sort_values("month")
        )

    openai_tools_categorized = load_json(f"{OUTPUT_PATH}/metadata/all_tools_categorized.json")
    openai_monthly = _monthly_web_rate(df, openai_tools_categorized)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=openai_monthly["month"],
        y=openai_monthly["cat_tooly_turns"],
        mode="lines+markers",
        name=f"OpenAI",
    ))

    for cai in ["claude", "grok", "deepseek"]:
        cai_df = load_whole_data_from_file_cai(fmt="pkl", platform=cai)
        tools_categorized = load_json(f"{OUTPUT_PATH}/{cai}/metadata/all_tools_categorized.json")
        monthly_tooly_turns = _monthly_web_rate(cai_df, tools_categorized)
        fig.add_trace(go.Scatter(
            x=monthly_tooly_turns["month"],
            y=monthly_tooly_turns["cat_tooly_turns"],
            mode="lines+markers",
            name=cai.capitalize(),
        ))


    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Turns (%)",
        xaxis=dict(
            type="date",
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "tooly_turns_rate_over_time_across_convais"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18), legend_pos=(0.9, 1.2))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")



def web_call_trend_over_time_by_model(df):
    df = df.copy()
    selected_models = ['gpt-4-1', 'gpt-4-1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-instant', 'gpt-5-mini', 'gpt-5-thinking', 'gpt-5-2', 'gpt-5-2-thinking', 'o3', 'o3-mini', 'text-davinci-002-render-sha']
    tools_categorized = load_json(f"{OUTPUT_PATH}/metadata/all_tools_categorized.json")
    tool_to_category = {
        tool: cat for cat, cat_tools in tools_categorized.items() for tool, id in cat_tools.items()
    }
    df["categories"] = df["tools"].apply(lambda x: [tool_to_category[t] for t in x])
    df["month"] = pd.to_datetime(df["month"])
    df["model"] = df["openai_models"].apply(_primary_model)
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
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 14), legend_pos=(0.8, 1.8))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def print_available_models(df):
    df = df.copy()
    df["model"] = df["openai_models"].apply(_primary_model)
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
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 10))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

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
    df["has_web_call"] = df["interactions"].apply(
        lambda x: isinstance(x, list) and any("web" in str(item) for item in x)
    )

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
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 10))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def policy_distribution():
    df = pd.read_csv(f"{OUTPUT_PATH}/metadata/web_calls_characterization.csv")
    web_df = load_web_data_from_file(fmt="pkl")
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
    primary_trigger = []
    secondary_triggers = []
    policy_samples = {}
    for i, row in df.iterrows():
        policy = json.loads(row["followed_web_policy"])
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
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18))
    fig.update_xaxes(tickangle=-45)
    fig.update_xaxes(tickfont=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")
    to_json(
        [
            policy_samples[policy]
            for policy in primary_rates["policy"].tolist()
            if policy in policy_samples
        ],
        f"{OUTPUT_PATH}/{CONF}/web_call_policy_characterization_samples.json",
    )


def policy_distribution_stacked_by_topics():
    df = pd.read_csv(f"{OUTPUT_PATH}/metadata/web_calls_characterization.csv")
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

    plot_rows = []
    for _, row in df.iterrows():
        if row["topic"] not in selected_topics:
            continue
        policy = json.loads(row["followed_web_policy"])
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
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16), legend_pos=(0.8, 1.4))
    fig.update_xaxes(tickangle=-45)
    fig.update_xaxes(tickfont=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def _prometheus_eval_base_dir(
    eval_model_name="Unbabel/M-Prometheus-14B",
    temperature="1.0",
    web_tool_type="openai",
):
    safe_eval_model_name = str(eval_model_name).replace("/", "__")
    if web_tool_type == "openai":
        return (
            f"{OUTPUT_PATH}/metadata/prometheus_evaluation/"
            f"{safe_eval_model_name}/{temperature}"
        )
    return (
        f"{OUTPUT_PATH}/metadata/prometheus_evaluation/"
        f"{safe_eval_model_name}/{temperature}"
    )


def _load_replay_eval_df(base_dir, llm_model_name):
    file_path = f"{base_dir}/{llm_model_name}.csv"
    if not os.path.exists(file_path):
        print(f"Missing evaluation file: {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path).copy()
    original_len = len(df)

    # Some CSVs may contain malformed split rows when prompts include raw '\r'
    # characters. Keep only rows with stable sample keys.
    required_key_cols = [
        col for col in ["result_key", "conv_id", "turn_id"] if col in df.columns
    ]
    if required_key_cols:
        valid_mask = pd.Series(True, index=df.index)
        for col in required_key_cols:
            valid_mask = valid_mask & df[col].notna() & (
                df[col].astype(str).str.strip() != ""
            )
        df = df[valid_mask].copy()
        malformed_rows = original_len - len(df)
        if malformed_rows:
            print(
                f"Dropped {malformed_rows} malformed rows from {file_path} "
                "(missing key columns)."
            )

    if "result_key" in df.columns:
        deduped_len = len(df)
        df = df.drop_duplicates(subset=["result_key"], keep="first").copy()
        duplicate_rows = deduped_len - len(df)
        if duplicate_rows:
            print(f"Dropped {duplicate_rows} duplicate rows by result_key.")

    return df


def replay_evaluations_prometheus(
    eval_model_name="Unbabel/M-Prometheus-14B",
    llm_model_name="gpt-5-mini-2025-08-07",
    temperature="1.0",
    web_tool_type="openai",
):
    base_dir = _prometheus_eval_base_dir(
        eval_model_name=eval_model_name,
        temperature=temperature,
        web_tool_type=web_tool_type,
    )
    df = _load_replay_eval_df(base_dir, llm_model_name)
    if len(df) == 0:
        return pd.DataFrame()

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
        return pd.DataFrame()

    likert_metrics = [
        "factuality_5likert",
        "relevance_5likert",
        "completeness_5likert",
    ]
    metric_label_map = {
        "factuality_5likert": "Factuality",
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
        "\nPrometheus replay evaluation average 5-Likert scores "
        f"({llm_model_name}, judged by {eval_model_name}):"
    )
    print(display_df.to_string(index=False))

    x_labels = metric_labels + ["Average"]
    fig = go.Figure()
    for _, row in summary_df.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["Response Mode"],
                x=x_labels,
                y=[row.get(label, float("nan")) for label in x_labels],
            )
        )

    fig.update_layout(
        title="Prometheus Replay Evaluation Summary",
        xaxis_title="Metric",
        yaxis_title="Average Score",
        barmode="group",
        yaxis=dict(range=[1, 5]),
    )
    output_dir = f"{OUTPUT_PATH}/{CONF}/replay_modes/{web_tool_type}/"
    os.makedirs(output_dir, exist_ok=True)
    safe_eval_model_name = str(eval_model_name).replace("/", "__")
    file_name = (
        f"replay_{llm_model_name}_prometheus_evaluations_"
        f"{safe_eval_model_name}_summary"
    )
    fig.write_html(f"{output_dir}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18), legend_pos=(0.5, 1.15))
    fig.update_layout(
        legend=dict(
            x=0.5,
            y=1.15,
            xanchor="center",
            yanchor="top",
            orientation="h",
        )
    )
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")
    return summary_df


def replay_call_outcome_summary_prometheus(
    eval_model_name="Unbabel/M-Prometheus-14B",
    llm_model_name="gpt-5-mini-2025-08-07",
    temperature="1.0",
    web_tool_type="openai",
):
    base_dir = _prometheus_eval_base_dir(
        eval_model_name=eval_model_name,
        temperature=temperature,
        web_tool_type=web_tool_type,
    )
    df = _load_replay_eval_df(base_dir, llm_model_name)
    print(len(df))
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
        "completeness_5likert",
    ]
    metric_label_map = {
        "factuality_5likert": "Factuality",
        "relevance_5likert": "Relevance",
        "completeness_5likert": "Completeness",
    }

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

    def _call_outcome(row, metric_names):
        return _strict_call_outcome(row, metric_names)

    def _plot_call_outcome_counts(plot_df, outcome_order, suffix, title):
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
        colors = {
            "Right Call": "#54A24B",
            "Over Call": "#E45756",
            "Under Call": "#F58518",
        }
        marker_colors = [colors[outcome] for outcome in outcome_order]
        inbar_text_size = 18

        outcome_fig = go.Figure()
        outcome_fig.add_trace(
            go.Bar(
                x=outcome_order,
                y=no_web_counts,
                name="No Web Call",
                text=text_no_web,
                textposition="inside",
                textfont=dict(color="black", size=inbar_text_size),
                marker_color=marker_colors,
                hovertemplate="%{x}<br>No web call: %{y}<extra></extra>",
                showlegend=False,
            )
        )
        outcome_fig.add_trace(
            go.Bar(
                x=outcome_order,
                y=web_counts,
                name="With Web Call",
                text=text_web,
                textposition="inside",
                textfont=dict(color="black", size=inbar_text_size),
                marker=dict(
                    color=marker_colors,
                    pattern=dict(
                        shape="/",
                        fgcolor="#000000",
                        size=8,
                        solidity=0.25,
                    ),
                ),
                hovertemplate="%{x}<br>With web call: %{y}<extra></extra>",
                showlegend=False,
            )
        )
        # Custom legend markers so legend encodes call type style (empty vs dashed)
        # rather than inheriting the first outcome color (green).
        outcome_fig.add_trace(
            go.Bar(
                x=[outcome_order[0]],
                y=[0],
                name="No Web Call",
                marker=dict(
                    color="rgba(0, 0, 0, 0)",
                    line=dict(color="#000000", width=1.5),
                ),
                hoverinfo="skip",
            )
        )
        outcome_fig.add_trace(
            go.Bar(
                x=[outcome_order[0]],
                y=[0],
                name="With Web Call",
                marker=dict(
                    color="rgba(0, 0, 0, 0)",
                    line=dict(color="#000000", width=1.5),
                    pattern=dict(
                        shape="/",
                        fgcolor="#000000",
                        size=8,
                        solidity=0.25,
                    ),
                ),
                hoverinfo="skip",
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
            uniformtext_minsize=inbar_text_size,
            uniformtext_mode="hide",
        )
        outcome_fig.update_yaxes(
            range=[0, max(max(total_counts) * 1.2, 1)]
        )
        safe_eval_model_name = str(eval_model_name).replace("/", "__")
        file_name = (
            f"replay_{llm_model_name}_prometheus_evaluations_{safe_eval_model_name}_"
            f"{decision_file_suffix}_call_outcome_{suffix}"
        )
        os.makedirs(f"{OUTPUT_PATH}/{CONF}/replay_modes/{web_tool_type}/", exist_ok=True)
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
                font=dict(color="#000000"),
            )
        )
        outcome_fig.write_image(f"{OUTPUT_PATH}/{CONF}/replay_modes/{web_tool_type}/{file_name}.pdf", format="pdf")

    def _plot_outcome_for_metrics(metric_names, suffix, title):
        plot_df = df.copy()
        plot_df["call_outcome"] = plot_df.apply(
            lambda row: _call_outcome(row, metric_names), axis=1
        )
        outcome_order = ["Right Call", "Over Call", "Under Call"]
        _plot_call_outcome_counts(
            plot_df,
            outcome_order,
            suffix,
            title,
        )

    def _plot_conservative_outcome_for_metrics(metric_names, suffix, title):
        plot_df = df.copy()
        plot_df["call_outcome"] = plot_df.apply(
            lambda row: _call_outcome(row, metric_names), axis=1
        )
        outcome_order = ["Right Call", "Over Call", "Under Call"]
        _plot_call_outcome_counts(
            plot_df,
            outcome_order,
            suffix,
            title,
        )

    _plot_conservative_outcome_for_metrics(
        likert_metrics,
        "5likert_conservative",
        "Prometheus 5-Likert Conservative Call Outcome",
    )
    for metric in likert_metrics:
        metric_label = metric_label_map.get(
            metric, metric.replace("_", " ").title()
        )
        metric_suffix = metric.replace("_5likert", "")
        _plot_conservative_outcome_for_metrics(
            [metric],
            f"5likert_{metric_suffix}_conservative",
            f"Prometheus 5-Likert {metric_label} Conservative Call Outcome",
        )


def replay_evaluations(
    eval_model_name="gpt-5.4-nano-2026-03-17",
    llm_model_name="gpt-5.4-nano-2026-03-17",
    temperature="0.0",
    web_tool_type="openai"
):
    if web_tool_type == "openai":
        base_dir = f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}"
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
        base_dir = f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}"
    else:
        base_dir = f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}"

    df = pd.read_csv(f"{base_dir}/{llm_model_name}.csv").copy()
    print(len(df))
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
    metric_label_map = {
        "factuality_5likert": "Factuality",
        "relevance_5likert": "Relevance",
        "completeness_5likert": "Completeness",
    }

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
            df, ["auto_called_web", "auto_web_call"]
        )
        decision_label = "Auto"
        decision_file_suffix = "auto"

    def _call_outcome(row, metric_names):
        return _strict_call_outcome(row, metric_names)

    def _plot_call_outcome_counts(plot_df, outcome_order, suffix, title):
        outcome_counts = (
            plot_df.groupby(["decision_uses_web", "call_outcome"])
            .size()
            .unstack(fill_value=0)
        )

        def _count(decision_uses_web, outcome):
            if decision_uses_web not in outcome_counts.index:
                return 0
            return int(outcome_counts.loc[decision_uses_web].get(outcome, 0))

        x_labels = ["No Web Call", "Web Called"]
        right_counts = [
            _count(False, "Right Call"),
            _count(True, "Right Call"),
        ]
        under_counts = [
            _count(False, "Under Call"),
            0,
        ]
        over_counts = [
            0,
            _count(True, "Over Call"),
        ]

        total_counts = [int((~plot_df["decision_uses_web"]).sum()), int(plot_df["decision_uses_web"].sum())]
        inbar_text_size = 24

        outcome_fig = go.Figure()
        outcome_fig.add_trace(
            go.Bar(
                x=x_labels,
                y=right_counts,
                name="Right Call",
                text=[str(count) if count else "" for count in right_counts],
                textposition="inside",
                textfont=dict(color="black", size=inbar_text_size),
                marker_color="#54A24B",
                hovertemplate="%{x}<br>Right Call: %{y}<extra></extra>",
            )
        )
        outcome_fig.add_trace(
            go.Bar(
                x=x_labels,
                y=under_counts,
                name="Under Call",
                text=[str(count) if count else "" for count in under_counts],
                textposition="inside",
                textfont=dict(color="black", size=inbar_text_size),
                marker_color="#F58518",
                hovertemplate="%{x}<br>Under Call: %{y}<extra></extra>",
            )
        )
        outcome_fig.add_trace(
            go.Bar(
                x=x_labels,
                y=over_counts,
                name="Over Call",
                text=[str(count) if count else "" for count in over_counts],
                textposition="inside",
                textfont=dict(color="black", size=inbar_text_size),
                marker_color="#E45756",
                hovertemplate="%{x}<br>Over Call: %{y}<extra></extra>",
            )
        )
        outcome_fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=total_counts,
                mode="text",
                text=[str(count) if count else "" for count in total_counts],
                textposition="top center",
                textfont=dict(color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        outcome_fig.update_layout(
            title=title,
            xaxis_title=f"{decision_label} Web Call Decision",
            yaxis_title="Samples",
            barmode="stack",
            uniformtext_minsize=inbar_text_size,
            uniformtext_mode="hide",
        )
        outcome_fig.update_yaxes(
            range=[0, max(max(total_counts) * 1.2, 1)]
        )
        file_name = (
            f"replay_{llm_model_name}_evaluations_{eval_model_name}_"
            f"{decision_file_suffix}_call_outcome_{suffix}"
        )
        os.makedirs(f"{OUTPUT_PATH}/{CONF}/replay_modes/{web_tool_type}/", exist_ok=True)
        # outcome_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{web_tool_type}/{file_name}.html")
        outcome_fig = with_paper_style(
            outcome_fig, config=styler(24, 24), legend_pos=(0.5, 1.15)
        )
        outcome_fig.update_layout(
            legend=dict(
                x=0.5,
                y=1.15,
                xanchor="center",
                yanchor="top",
                orientation="h",
                font=dict(color="#000000"),
            )
        )
        outcome_fig.write_image(f"{OUTPUT_PATH}/{CONF}/replay_modes/{web_tool_type}/{file_name}.pdf", format="pdf")

    def _plot_outcome_for_metrics(metric_names, suffix, title):
        plot_df = df.copy()
        plot_df["call_outcome"] = plot_df.apply(
            lambda row: _call_outcome(row, metric_names), axis=1
        )
        outcome_order = ["Right Call", "Over Call", "Under Call"]
        _plot_call_outcome_counts(
            plot_df,
            outcome_order,
            suffix,
            title,
        )

    def _plot_conservative_outcome_for_metrics(metric_names, suffix, title):
        plot_df = df.copy()
        plot_df["call_outcome"] = plot_df.apply(
            lambda row: _call_outcome(row, metric_names), axis=1
        )
        outcome_order = ["Right Call", "Over Call", "Under Call"]
        _plot_call_outcome_counts(
            plot_df,
            outcome_order,
            suffix,
            title,
        )

    _plot_conservative_outcome_for_metrics(
        likert_metrics,
        "5likert",
        "5-Likert Call Outcome",
    )
    for metric in likert_metrics:
        metric_label = metric_label_map.get(
            metric, metric.replace("_", " ").title()
        )
        metric_suffix = metric.replace("_5likert", "")
        _plot_conservative_outcome_for_metrics(
            [metric],
            f"5likert_{metric_suffix}",
            f"5-Likert {metric_label} Call Outcome",
        )


def replay_call_outcome_venn_diagram(
    eval_model_name="gpt-5.4-nano-2026-03-17",
    llm_model_name="gpt-5.4-nano-2026-03-17",
    temperature="0.0",
):
    tool_configs = [
        {
            "tool": "openai",
            "label": "OpenAI",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": llm_model_name,
        },
        {
            "tool": "tavily",
            "label": "Tavily",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": f"{llm_model_name}_responses_url_mcp-tavily",
        },
        {
            "tool": "serp",
            "label": "SerpAPI",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": f"{llm_model_name}_responses_url_mcp-serp",
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
    merge_keys = ["conv_id", "turn_id"]

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
    num_tools = len(tool_order)
    if len(merged_df) == 0:
        print(f"No common samples found across all {num_tools} tools.")
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

    print(f"Common samples across all {num_tools} tools: {len(merged_df)}")
    print(
        f"Samples where all {num_tools} tools called web in auto mode: "
        f"{all_tools_count}"
    )
    print(f"Samples where no tool called web in auto mode: {no_tool_count}")
    print("\nAuto web-call exact intersections:")
    print(summary_df.to_string(index=False))

    output_dir = f"{OUTPUT_PATH}/{CONF}/replay_modes/"
    os.makedirs(output_dir, exist_ok=True)
    file_name = (
        f"replay_{llm_model_name}_auto_web_call_venn_"
        f"{eval_model_name}"
    )
    # summary_df.to_csv(f"{output_dir}/{file_name}_summary.csv", index=False)
    # to_json(
    #     summary_df.to_dict(orient="records"),
    #     f"{output_dir}/{file_name}_summary.json",
    # )

    openai_tool = "openai"
    if openai_tool not in merged_df.columns:
        print("Missing OpenAI auto web-call column.")
        return summary_df

    other_tools = [tool for tool in tool_order if tool != openai_tool]
    other_tool_count = len(other_tools)

    openai_called_mask = merged_df[openai_tool].astype(bool)
    openai_not_called_mask = ~openai_called_mask
    other_called_counts = (
        merged_df[other_tools].astype(bool).sum(axis=1).astype(int)
        if other_tools
        else pd.Series(0, index=merged_df.index, dtype=int)
    )
    other_not_called_counts = other_tool_count - other_called_counts
    segment_order = list(range(other_tool_count, -1, -1))

    openai_called_distribution = {
        match_count: int(
            (openai_called_mask & (other_called_counts == match_count)).sum()
        )
        for match_count in segment_order
    }
    openai_not_called_distribution = {
        match_count: int(
            (openai_not_called_mask & (other_not_called_counts == match_count)).sum()
        )
        for match_count in segment_order
    }

    print("\nOpenAI-aligned web-call distribution:")
    for match_count in segment_order:
        print(
            f"OpenAI called web + {match_count} of {other_tool_count} "
            f"other tools called web: {openai_called_distribution[match_count]}"
        )
    for match_count in segment_order:
        print(
            f"OpenAI did not call web + {match_count} of {other_tool_count} "
            f"other tools did not call web: "
            f"{openai_not_called_distribution[match_count]}"
        )

    color_scale = qualitative.Plotly
    if len(color_scale) < len(segment_order):
        multiplier = (len(segment_order) // len(color_scale)) + 1
        color_scale = (color_scale * multiplier)[: len(segment_order)]

    stacked_fig = go.Figure()
    y_labels = ["OpenAI<br>Not Called<br>Web Search", "OpenAI<br>Called<br>Web Search"]
    for idx, match_count in enumerate(segment_order):
        called_value = openai_called_distribution[match_count]
        not_called_value = openai_not_called_distribution[match_count]
        stacked_fig.add_trace(
            go.Bar(
                y=y_labels,
                x=[not_called_value, called_value],
                orientation="h",
                name=f"{match_count} of {other_tool_count} other tools",
                marker_color=color_scale[idx],
                text=[
                    str(not_called_value) if not_called_value else "",
                    str(called_value) if called_value else "",
                ],
                textposition="inside",
                # customdata=[
                #     (
                #         f"{match_count} of {other_tool_count} other tools "
                #         "called web"
                #     ),
                #     (
                #         f"{match_count} of {other_tool_count} other tools "
                #         "did not call web"
                #     ),
                # ],
                hovertemplate=(
                    "%{y}<br>%{customdata}<br>Samples: %{x}<extra></extra>"
                ),
            )
        )

    stacked_fig.update_layout(
        barmode="stack",
        xaxis_title="Samples",
        yaxis_title="",
        legend_title=f"Matching Tools",
        margin=dict(l=5, r=20, t=100, b=80),
    )
    stacked_fig = with_paper_style(
        stacked_fig, config=styler(20, 18)
    )
    stacked_fig.update_layout(
        margin=dict(l=5, r=20, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    # Keep labels compact so Plotly does not auto-reserve wide left space.
    # stacked_fig.update_yaxes(
    #     categoryorder="array",
    #     categoryarray=y_labels[::-1],
    #     automargin=False,
    #     ticklabelposition="inside",
    #     ticklabelstandoff=0,
    #     ticks="",
    # )
    # stacked_fig.write_html(f"{output_dir}/{file_name}.html")
    stacked_fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

    agreement_file_name = f"{file_name}_agreement_counts"
    # stacked_fig.write_html(f"{output_dir}/{agreement_file_name}.html")
    stacked_fig.write_image(
        f"{output_dir}/{agreement_file_name}.pdf", format="pdf"
    )

    return summary_df


def replay_all_tools_web_call_metric_comparison(
    eval_model_name="gpt-5.4-nano-2026-03-17",
    llm_model_name="gpt-5.4-nano-2026-03-17",
    temperature="0.0",
):
    tool_configs = [
        {
            "tool": "openai",
            "label": "OpenAI",
            "base_dir": (
                f"{OUTPUT_PATH}/metadata/preference_evaluation/"
                f"{eval_model_name}/{temperature}"
            ),
            "file_name": llm_model_name,
        },
        {
            "tool": "serp",
            "label": "SerpAPI",
            "base_dir": (
                f"{OUTPUT_PATH}/metadata/preference_evaluation/"
                f"{eval_model_name}/{temperature}"
            ),
            "file_name": f"{llm_model_name}_responses_url_mcp-serp",
        },
        {
            "tool": "perplexity",
            "label": "Perplexity",
            "base_dir": (
                f"{OUTPUT_PATH}/metadata/preference_evaluation/"
                f"{eval_model_name}/{temperature}"
            ),
            "file_name": f"{llm_model_name}_responses_url_mcp-perplexity",
        },
        {
            "tool": "brave",
            "label": "Brave",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": f"{llm_model_name}_responses_url_mcp-brave",
        },
        {
            "tool": "tavily",
            "label": "Tavily",
            "base_dir": f"{OUTPUT_PATH}/metadata/preference_evaluation/{eval_model_name}/{temperature}",
            "file_name": f"{llm_model_name}_responses_url_mcp-tavily",
        },
    ]
    merge_keys = ["conv_id", "turn_id"]
    likert_metrics = [
        "factuality_5likert",
        "relevance_5likert",
        "completeness_5likert",
    ]
    metric_label_map = {
        "factuality_5likert": "Factuality",
        "relevance_5likert": "Relevance",
        "completeness_5likert": "Completeness",
    }
    metric_labels = [
        metric_label_map.get(metric, metric.replace("_", " ").title())
        for metric in likert_metrics
    ]
    metric_score_cols = [f"auto_{metric}_score" for metric in likert_metrics]

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

        web_call_col = f"{config['tool']}_uses_web"
        df[web_call_col] = _web_call_flag_series(
            df, ["auto_called_web", "auto_web_call"]
        )

        for score_col in metric_score_cols:
            if score_col not in df.columns:
                df[score_col] = float("nan")

        rename_map = {
            score_col: f"{config['tool']}__{score_col}"
            for score_col in metric_score_cols
        }
        selected_cols = merge_keys + [web_call_col] + metric_score_cols
        return df[selected_cols].rename(columns=rename_map)

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

    num_tools = len(tool_configs)
    if len(merged_df) == 0:
        print(f"No common samples found across all {num_tools} tools.")
        return pd.DataFrame()

    web_call_cols = [f"{config['tool']}_uses_web" for config in tool_configs]
    all_web_df = merged_df[merged_df[web_call_cols].all(axis=1)].copy()
    if len(all_web_df) == 0:
        print(f"No samples where all {num_tools} tools call web in auto mode.")
        return pd.DataFrame()

    summary_rows = []
    for config in tool_configs:
        tool = config["tool"]
        row = {"Tool": config["label"]}
        metric_values = []
        for metric, metric_label in zip(likert_metrics, metric_labels):
            score_col = f"{tool}__auto_{metric}_score"
            score = pd.to_numeric(all_web_df[score_col], errors="coerce").mean()
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

    print(
        f"Samples where all {num_tools} tools call web in auto mode: "
        f"{len(all_web_df)}"
    )
    print("\nAll-tools-web auto response quality summary:")
    print(display_df.to_string(index=False))

    x_labels = [config["label"] for config in tool_configs]
    y_columns = metric_labels + ["Average"]
    plot_colors = qualitative.Plotly
    fig = go.Figure()
    for idx, y_col in enumerate(y_columns):
        fig.add_trace(
            go.Bar(
                name=y_col,
                x=x_labels,
                y=summary_df[y_col],
                marker_color=plot_colors[idx % len(plot_colors)],
                text=[
                    f"{value:.2f}" if pd.notna(value) else ""
                    for value in summary_df[y_col]
                ],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>" + y_col + ": %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        # title=f"Auto Response Quality on All-Tools-Web Samples ({num_tools} tools)",
        xaxis_title="Tool",
        yaxis_title="Average Score",
        barmode="group",
    )
    fig.update_yaxes(range=[1, 5])

    output_dir = f"{OUTPUT_PATH}/{CONF}/replay_modes/"
    os.makedirs(output_dir, exist_ok=True)
    safe_eval_model_name = str(eval_model_name).replace("/", "__")
    file_name = (
        f"replay_{llm_model_name}_all_tools_web_call_quality_"
        f"{safe_eval_model_name}"
    )
    # summary_df.to_csv(f"{output_dir}/{file_name}_summary.csv", index=False)
    # to_json(
    #     summary_df.to_dict(orient="records"),
    #     f"{output_dir}/{file_name}_summary.json",
    # )

    # fig.write_html(f"{output_dir}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(20, 16), legend_pos=(0.5, 1.15))
    fig.update_layout(
        legend=dict(
            x=0.5,
            y=1.15,
            xanchor="center",
            yanchor="top",
            orientation="h",
        ),
        margin=dict(l=30, r=20, t=70, b=45),
    )
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")
    return summary_df


def _load_tool_intent_input(input_path=None, input_fmt="parquet"):
    if input_path is None:
        return load_web_data_from_file(fmt=input_fmt).copy()

    ext = os.path.splitext(str(input_path))[1].lower()
    if ext == ".csv":
        return pd.read_csv(input_path)
    if ext == ".pkl":
        return pd.read_pickle(input_path)
    if ext == ".parquet":
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input file type: {ext}")


def _normalise_output_base(output_base, model_name):
    if output_base is None:
        safe_model_name = str(model_name).replace("/", "_")
        output_base = (
            f"{OUTPUT_PATH}/metadata/web_call_tool_intent_from_thoughts_"
            f"{safe_model_name}"
        )

    root, ext = os.path.splitext(str(output_base))
    if ext in {".csv", ".pkl", ".json"}:
        return root
    return str(output_base)


def _tool_intent_record_key(record):
    conv_id = str(record.get("conv_id", "")).strip()
    turn_id = str(record.get("turn_id", "")).strip()
    source_index = str(record.get("source_index", "")).strip()
    if conv_id or turn_id:
        return (conv_id, turn_id, source_index)
    return ("", "", source_index)


def _load_existing_tool_intent_records(output_base):
    csv_path = f"{output_base}.csv"
    if not os.path.exists(csv_path):
        return [], set()

    existing_df = pd.read_csv(csv_path)
    records = existing_df.to_dict(orient="records")
    keys = {_tool_intent_record_key(record) for record in records}
    return records, keys


def _save_tool_intent_records(records, output_base):
    output_dir = os.path.dirname(output_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    safe_records = [
        {key: _json_safe_value(value) for key, value in record.items()}
        for record in records
    ]
    results_df = pd.DataFrame(safe_records)
    results_df.to_csv(f"{output_base}.csv", index=False)
    results_df.to_pickle(f"{output_base}.pkl")
    to_json(safe_records, f"{output_base}.json")
    return results_df


def _run_tool_intent_judge(
    client,
    model_name,
    user_message,
    thinking,
    temperature=0.0,
    max_output_tokens=256,
):
    response = client.responses.create(
        model=model_name,
        tools=[],
        tool_choice="none",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_TOOL_INTENT},
            {
                "role": "user",
                "content": USER_PROMPT_TOOL_INTENT.format(
                    user_message=user_message,
                    thinking=thinking,
                ),
            },
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    raw_text = response.output_text
    return {
        "raw_judgment": raw_text,
        "parsed_judgment": _parse_eval_json(raw_text),
    }


def classify_web_call_tool_intent_from_thoughts(
    model_name="gpt-4o-mini",
    input_path=None,
    input_fmt="parquet",
    output_base=None,
    sample_size=None,
    random_state=0,
    only_english=True,
    require_conv_starter=True,
    require_single_user_message=True,
    fallback_to_thoughts_column=False,
    resume=True,
    save_every=25,
    temperature=0.0,
    max_output_tokens=256,
    max_user_message_chars=8000,
    max_thinking_chars=20000,
    make_pie_chart=True,
):
    """
    Find web-call samples with reasoning traces and classify the search intent.

    The judge labels each sample as:
    - Verified Prior Knowledge
    - Acquired New Information
    - Mixed

    When `turn_msgs` is available, the function uses thoughts before the first
    assistant-to-web message/search query. For query-reformulation data, it uses
    only the first `thoughts_list` entry before the first web query. The aggregate
    full-turn `thoughts` column is used only when `fallback_to_thoughts_column`
    is explicitly enabled.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    output_base = _normalise_output_base(output_base, model_name)
    df = _load_tool_intent_input(input_path=input_path, input_fmt=input_fmt)

    original_len = len(df)
    if only_english and "language" in df.columns:
        df = df[df["language"] == "en"].copy()
    if require_conv_starter and "conv_starter" in df.columns:
        df = df[df["conv_starter"] == 1].copy()
    if require_single_user_message and "user_msg_history" in df.columns:
        df = df[
            df["user_msg_history"].apply(
                lambda value: len(_as_list_value(value)) == 1
            )
        ].copy()
    print(f"Loaded {len(df)} samples after filtering out of {original_len}.")

    samples = []
    for source_index, row in df.iterrows():
        if not _row_has_web_call(row):
            continue

        thinking = _thoughts_before_first_web_call(row)
        thinking_source = "turn_msgs_before_first_web_call" if thinking else ""
        if not thinking:
            thinking = _thoughts_before_first_web_query(row)
            thinking_source = "thoughts_list_before_first_web_query" if thinking else ""
        if not thinking and fallback_to_thoughts_column:
            thinking = _stringify_nested_text(row.get("thoughts", ""))
            thinking_source = "thoughts_column_full_turn" if thinking else ""
        if not thinking:
            continue

        user_message = _latest_user_message(row)
        if not user_message:
            continue

        samples.append(
            {
                "source_index": source_index,
                "user_id": row.get("user_id"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "conv_starter": row.get("conv_starter"),
                "topic": row.get("topic"),
                "language": row.get("language"),
                "time": row.get("time"),
                "user_message": user_message,
                "thinking": thinking,
                "thinking_source": thinking_source,
            }
        )

    samples_df = pd.DataFrame(samples)
    print(f"Found {len(samples_df)} web-call samples with thoughts.")
    if len(samples_df) == 0:
        return samples_df

    if sample_size is not None and sample_size < len(samples_df):
        samples_df = samples_df.sample(
            n=sample_size,
            random_state=random_state,
        ).reset_index(drop=True)
        print(f"Judging sampled subset of {len(samples_df)} rows.")

    records = []
    completed_keys = set()
    if resume:
        records, completed_keys = _load_existing_tool_intent_records(output_base)
        if records:
            print(f"Loaded {len(records)} existing judgments from {output_base}.csv.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for _, sample in tqdm(samples_df.iterrows(), total=len(samples_df)):
        sample_record = sample.to_dict()
        sample_key = _tool_intent_record_key(sample_record)
        if sample_key in completed_keys:
            continue

        record = {
            **sample_record,
            "judge_model": model_name,
            "temperature": temperature,
        }
        try:
            eval_result = _run_tool_intent_judge(
                client=client,
                model_name=model_name,
                user_message=_trim_text(
                    sample_record["user_message"],
                    max_user_message_chars,
                ),
                thinking=_trim_text(
                    sample_record["thinking"],
                    max_thinking_chars,
                ),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            parsed = eval_result["parsed_judgment"]
            if not isinstance(parsed, dict):
                parsed = {}
            record["tool_intent_label"] = parsed.get("label")
            record["tool_intent_reasoning"] = parsed.get("reasoning")
            record["tool_intent_raw_judgment"] = eval_result["raw_judgment"]
            record["tool_intent_status"] = "ok" if parsed else "parse_failed"
            record["tool_intent_error"] = ""
        except Exception as e:
            print(
                "classify_web_call_tool_intent_from_thoughts",
                sample_record.get("conv_id"),
                sample_record.get("turn_id"),
                e,
            )
            record["tool_intent_label"] = ""
            record["tool_intent_reasoning"] = ""
            record["tool_intent_raw_judgment"] = ""
            record["tool_intent_status"] = "failed"
            record["tool_intent_error"] = str(e)

        records.append(record)
        completed_keys.add(sample_key)
        if save_every and len(records) % save_every == 0:
            _save_tool_intent_records(records, output_base)

    results_df = _save_tool_intent_records(records, output_base)
    if "tool_intent_label" in results_df.columns:
        print("\nTool-intent label counts:")
        print(results_df["tool_intent_label"].value_counts(dropna=False).to_string())
    if make_pie_chart:
        plot_web_call_tool_intent_distribution(
            results_df=results_df,
            output_base=output_base,
            model_name=model_name,
        )
    print(f"Saved judgments to {output_base}.csv/.pkl/.json")
    return results_df


def _load_tool_intent_results(input_path=None, output_base=None, model_name="gpt-4o-mini"):
    if input_path is None:
        output_base = _normalise_output_base(output_base, model_name)
        pkl_path = f"{output_base}.pkl"
        csv_path = f"{output_base}.csv"
        if os.path.exists(pkl_path):
            return pd.read_pickle(pkl_path)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        print(f"Missing tool-intent results at {pkl_path} or {csv_path}.")
        return pd.DataFrame()

    ext = os.path.splitext(str(input_path))[1].lower()
    if ext == ".csv":
        return pd.read_csv(input_path)
    if ext == ".pkl":
        return pd.read_pickle(input_path)
    if ext == ".parquet":
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input file type: {ext}")


def plot_web_call_tool_intent_distribution(
    results_df=None,
    input_path=None,
    output_base=None,
    model_name="gpt-4o-mini",
    file_name=None,
):
    if results_df is None:
        results_df = _load_tool_intent_results(
            input_path=input_path,
            output_base=output_base,
            model_name=model_name,
        )
    else:
        results_df = results_df.copy()

    if results_df.empty or "tool_intent_label" not in results_df.columns:
        print("No tool-intent labels found to plot.")
        return pd.DataFrame()

    plot_df = results_df.copy()
    if "tool_intent_status" in plot_df.columns:
        plot_df = plot_df[plot_df["tool_intent_status"] == "ok"].copy()

    label_order = [
        "Verified Prior Knowledge",
        "Acquired New Information",
        "Mixed",
    ]
    labels = plot_df["tool_intent_label"].fillna("").astype(str).str.strip()
    labels = labels[labels != ""]
    if labels.empty:
        print("No non-empty tool-intent labels found to plot.")
        return pd.DataFrame()

    plot_df = (
        labels.value_counts()
        .rename_axis("tool_intent_label")
        .reset_index(name="count")
    )
    plot_df["percentage"] = plot_df["count"] / plot_df["count"].sum()
    plot_df["sort_order"] = plot_df["tool_intent_label"].apply(
        lambda label: (
            label_order.index(label)
            if label in label_order
            else len(label_order)
        )
    )
    plot_df = plot_df.sort_values(
        ["sort_order", "tool_intent_label"]
    ).drop(columns=["sort_order"])

    colors = {
        "Verified Prior Knowledge": "#4C78A8",
        "Acquired New Information": "#F58518",
        "Mixed": "#54A24B",
    }
    marker_colors = [
        colors.get(label, "#BDBDBD")
        for label in plot_df["tool_intent_label"]
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=plot_df["tool_intent_label"],
                values=plot_df["count"],
                customdata=plot_df["percentage"],
                textinfo="label+percent",
                textposition="inside",
                textfont=dict(size=20),
                sort=False,
                hole=0.0,
                showlegend=False,
                marker=dict(colors=marker_colors),
                hovertemplate=(
                    "%{label}<br>"
                    "Count: %{value}<br>"
                    "Share: %{customdata:.1%}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Why the Assistant Called the Web",
        width=900,
        height=700,
        margin=dict(t=80, b=40, l=40, r=40),
        showlegend=False,
    )

    if file_name is None:
        safe_model_name = str(model_name).replace("/", "_")
        file_name = f"web_call_tool_intent_distribution_{safe_model_name}"
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f"{output_dir}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(22, 16))
    try:
        fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")
    except Exception as e:
        first_error_line = str(e).strip().splitlines()[0] if str(e).strip() else e
        print(f"Could not write PDF pie chart; HTML was saved. {first_error_line}")

    print("\nTool-intent distribution:")
    print(plot_df.to_string(index=False))
    return plot_df


def subset_selection_for_policy_evaluation_by_human():
    df = pd.read_csv(f"{OUTPUT_PATH}/metadata/web_calls_characterization.csv").reset_index()
    subset = df.sample(100)
    subset.to_csv(f"{OUTPUT_PATH}/metadata/web_calls_characterization_subset_for_human_eval.csv", index=False)


def count_model_used(web_df):
    # Count model usage across all web-call turns. A turn can list multiple models,
    # so we report both total mentions and the final/primary model per turn.
    df = web_df.copy()

    if "openai_models" not in df.columns:
        print("Column `openai_models` not found.")
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

    df["openai_models"] = df["openai_models"].apply(_ensure_model_list)
    df["primary_model"] = df["openai_models"].apply(_primary_model)

    all_model_counts = (
        df["openai_models"]
        .explode()
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="count")
    )

#     Model mentions across `openai_models`:
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

    print("\nModel mentions across `openai_models`:")
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
            f"{OUTPUT_PATH}/metadata/preference_evaluation/"
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

    output_base = f"{OUTPUT_PATH}/metadata/query_type_annotation_comparison"
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
    # full_df = load_whole_data_from_file(fmt="pkl")
    # print("# all turns:", len(full_df))
    # web_df = load_web_data_from_file(fmt="pkl")
    # print("# turns with web call:", len(web_df))

    # web_call_trend_over_time(full_df)
    # web_call_trend_over_time_all_convai(full_df)
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

    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-serp", eval_model_name="gpt-5.4-mini", web_tool_type="serp")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-serp", eval_model_name="gpt-5.4-mini", web_tool_type="serp")
    
    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-brave", eval_model_name="gpt-5.4-mini", web_tool_type="brave")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-brave", eval_model_name="gpt-5.4-mini", web_tool_type="brave")
    
    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-tavily", eval_model_name="gpt-5.4-mini", web_tool_type="tavily")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-tavily", eval_model_name="gpt-5.4-mini", web_tool_type="tavily")
    
    # replay_evaluations(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-perplexity", eval_model_name="gpt-5.4-mini", web_tool_type="perplexity")
    # replay_call_outcome_summary(llm_model_name="gpt-5-mini-2025-08-07_responses_url_mcp-perplexity", eval_model_name="gpt-5.4-mini", web_tool_type="perplexity")
    
    replay_call_outcome_venn_diagram(llm_model_name="gpt-5-mini-2025-08-07", eval_model_name="gpt-5.4-mini")
    # replay_all_tools_web_call_metric_comparison(llm_model_name="gpt-5-mini-2025-08-07", eval_model_name="gpt-5.4-mini")

    # count_model_used(web_df)
    # df_creaton_for_annotation()

    # classify_web_call_tool_intent_from_thoughts()
    # plot_web_call_tool_intent_distribution()
    pass
