import os
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
import json
import ast
import argparse
import logging
from tqdm import tqdm
from openai import OpenAI
import tiktoken
from utils import *
from data_extraction import load_web_data_from_file


SYSTEM_PROMPT = """
You are an expert AI Data Annotator specializing in LLM tool usage and agentic reasoning.

Your task is to classify why a web search tool (`web.run`) was triggered for a given user message along with the reasoning traces.

### TAXONOMY OF SEARCH TRIGGERS
Select exactly ONE Primary Trigger and optionally any Secondary Triggers:

1. Volatile/Temporal Information — Time-sensitive or frequently changing info (e.g., news, weather, prices, sports, policies, releases).
2. Unfamiliar Term/Typo — Rare, ambiguous, or possibly misspelled terms requiring lookup.
3. High-Investment Recommendation — Decisions involving significant time, money, or commitment (e.g., travel, purchases, services).
4. Attribution/Sourcing Needed — Requires verifiable sources, citations, quotes, or links.
5. External Reference — Mentions a specific external resource not included in the prompt (e.g., URL, paper, dataset).
6. Low Confidence/Niche Fact — Obscure, highly specific, or emerging topics with high hallucination risk.
7. High-Stakes Accuracy — Medical, legal, or financial queries where errors could cause harm.
8. User Verification — User asks to confirm, validate, or fact-check information.
9. Explicit Command — User explicitly asks to search, browse, or check online.
10. None of the Above — No clear trigger applies.

### INSTRUCTIONS
- Select EXACTLY ONE Primary Trigger.
- Add Secondary Triggers only if clearly justified (avoid overuse).
- Prefer the most direct cause of the search, not indirect context.
- Keep explanations concise (1–2 sentences, no fluff).

### OUTPUT FORMAT (JSON ONLY)
{{
  "primary_trigger": "<exact taxonomy label>",
  "secondary_triggers": ["<optional>", "<optional>"],
  "explanation": "<brief reasoning>"
}}
"""


USER_PROMPT = """
Classify why a web search (`web.run`) was triggered.

### INPUT
User Message History:
{PROMPT}

Model Reasoning:
{THOUGHTS}

### OUTPUT FORMAT (JSON ONLY)
{{
  "primary_trigger": "<exact taxonomy label>",
  "secondary_triggers": ["<optional>", "<optional>"],
  "explanation": "<brief reasoning>"
}}
"""

context_window_dict = {
    "gpt-4o-mini": 128000
}


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


def estimate_tokens(text, model_name):
    if not text:
        return 0

    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    # Fallback when tiktoken is unavailable.
    return max(1, len(text) // 4)


def trim_text_to_token_budget(text, token_budget, model_name):
    text = str(text or "").strip()
    if token_budget <= 0 or not text:
        return ""

    if estimate_tokens(text, model_name) <= token_budget:
        return text

    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        trimmed = encoding.decode(tokens[-token_budget:])
        return trimmed.strip()

    approx_chars = token_budget * 4
    return text[-approx_chars:].strip()


def keep_recent_history_within_budget(history, token_budget, model_name):
    history = history if isinstance(history, list) else [history]
    kept_messages = []
    used_tokens = 0

    for item in reversed(history):
        text = str(item).strip()
        if not text:
            continue

        item_tokens = estimate_tokens(text, model_name)
        if not kept_messages:
            # Always keep the most recent message, even if it must be trimmed.
            if item_tokens > token_budget:
                text = trim_text_to_token_budget(text, token_budget, model_name)
                kept_messages.append(text)
                break
            kept_messages.append(text)
            used_tokens += item_tokens
            continue

        if used_tokens + item_tokens > token_budget:
            break

        kept_messages.append(text)
        used_tokens += item_tokens

    kept_messages.reverse()
    return "\n".join(kept_messages).strip()


def build_bounded_user_prompt(
    model_name,
    user_msg_history,
    thoughts,
    max_output_tokens,
    context_window,
    reserved_prompt_tokens,
    max_thought_tokens,
):
    available_prompt_tokens = max(
        1024, context_window - max_output_tokens - reserved_prompt_tokens
    )
    prompt_shell = USER_PROMPT.format(THOUGHTS="", PROMPT="")
    shell_tokens = estimate_tokens(prompt_shell, model_name)
    content_budget = max(256, available_prompt_tokens - shell_tokens)

    thoughts_budget = min(max_thought_tokens, int(content_budget * 0.4))
    trimmed_thoughts = trim_text_to_token_budget(thoughts, thoughts_budget, model_name)
    thoughts_tokens = estimate_tokens(trimmed_thoughts, model_name)

    history_budget = max(256, content_budget - thoughts_tokens)
    bounded_history = keep_recent_history_within_budget(
        user_msg_history, history_budget, model_name
    )

    filled_prompt = USER_PROMPT.format(
        THOUGHTS=trimmed_thoughts,
        PROMPT=bounded_history,
    )

    # Final guard: if the template overhead was underestimated, keep shrinking history first.
    while estimate_tokens(filled_prompt, model_name) > available_prompt_tokens:
        history_budget = int(history_budget * 0.85)
        if history_budget < 128:
            trimmed_thoughts = trim_text_to_token_budget(
                trimmed_thoughts, max(128, int(thoughts_tokens * 0.85)), model_name
            )
            thoughts_tokens = estimate_tokens(trimmed_thoughts, model_name)
        bounded_history = keep_recent_history_within_budget(
            user_msg_history, history_budget, model_name
        )
        filled_prompt = USER_PROMPT.format(
            THOUGHTS=trimmed_thoughts,
            PROMPT=bounded_history,
        )

        if history_budget < 64 and thoughts_tokens <= 128:
            break

    return filled_prompt


def openai_inference(args, df):
    model_name = args.model_name
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    new_df = df[
        (df["conv_starter"] == 1)
        & (df["user_msg_history"].apply(lambda x: len(x) == 1))
        & (df["language"] == "en")
    ].copy().reset_index(drop=True)
    followed_web_policy = [""] * len(new_df)
    print(f"Evaluating {len(new_df)} samples out of {len(df)} ...")

    for i, row in tqdm(new_df.iterrows()):
        try:
            # user_msg_history = "\n".join([str(x) for x in row["user_msg_history"]]).strip()
            thoughts = row["thoughts"]
            filled_prompt = build_bounded_user_prompt(
                model_name=model_name,
                user_msg_history=row["user_msg_history"],
                thoughts=thoughts,
                max_output_tokens=max_new_tokens,
                context_window=context_window_dict[model_name],
                reserved_prompt_tokens=args.reserved_prompt_tokens,
                max_thought_tokens=args.max_thought_tokens,
            )
            msg = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": filled_prompt},
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=msg,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            followed_web_policy[i] = text.strip()
        except Exception as e:
            print(i, e)
            continue

    new_df["followed_web_policy"] = followed_web_policy
    new_df[
        [
            "user_id",
            "conv_id",
            "turn_id",
            "topic",
            "language",
            "followed_web_policy",
            "thoughts",
            "time",
        ]
    ].to_csv(f"{OUTPUT_PATH}/metadata/web_calls_characterization.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please provide information")
    parser.add_argument(
        "-mn",
        "--model_name",
        help="Model name",
        default="gpt-4o-mini",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--temperature",
        help="Temperature",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch Size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-mnt",
        "--max_new_tokens",
        help="Max New Tokens",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--context_window",
        help="Approximate total context window for the selected OpenAI model",
        default=128000,
        type=int,
    )
    parser.add_argument(
        "--reserved_prompt_tokens",
        help="Prompt-space reserved for system text and safety margin",
        default=4000,
        type=int,
    )
    parser.add_argument(
        "--max_thought_tokens",
        help="Cap for reasoning trace tokens before trimming",
        default=12000,
        type=int,
    )
    args = parser.parse_args()
    print(args)

    df = load_web_data_from_file(fmt="parquet")
    openai_inference(args, df)
