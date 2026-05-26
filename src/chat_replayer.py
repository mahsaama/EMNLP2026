import json
import ast
from collections import Counter

from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI
from utils import *
import pandas as pd
from data_extraction import load_whole_data_from_file, load_web_data_from_file
from tqdm import tqdm
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ANNOTATIONS_TURNS_PATH = (
    "./outputs/Annotations_Turns_all.csv"
)
ANNOTATION_REQUIRED_COLUMNS = {
    "conv_id",
    "personal_presence",
    "special_category_presence",
}


model_replacements = {
    "gpt-5-1": "gpt-5.1-2025-11-13",
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return [value]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    return [value]


def _history_turn_depth(history_depth):
    if history_depth is None:
        return None
    return max(history_depth // 2, 0)


def _clean_messages(messages):
    return [str(msg).strip() for msg in _as_list(messages) if str(msg).strip()]


def _has_exact_history_depth(row, prior_turns):
    user_prompts = _clean_messages(row["user_msg_history"])
    assistant_prompts = _clean_messages(row["assistant_msg_history"])
    return (
        len(user_prompts) == prior_turns + 1
        and len(assistant_prompts) >= prior_turns
    )


def _safe_annotation_conv_ids(annotation_path=ANNOTATIONS_TURNS_PATH):
    annotations = pd.read_csv(
        annotation_path,
        usecols=lambda column: column in ANNOTATION_REQUIRED_COLUMNS,
        dtype=str,
    )
    missing_columns = ANNOTATION_REQUIRED_COLUMNS.difference(annotations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"annotation file is missing required columns: {missing}")

    annotations = annotations.dropna(subset=["conv_id"])
    is_safe = (
        annotations["personal_presence"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("no")
        & annotations["special_category_presence"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("no")
    )
    safe_by_conv = is_safe.groupby(annotations["conv_id"].astype(str)).all()
    return set(safe_by_conv[safe_by_conv].index)


def _filter_to_safe_conversations(df, safe_conv_ids):
    return df[df["conv_id"].astype(str).isin(safe_conv_ids)].copy()


def filter_df_for_history(
    history_depth=0,
    samples_per_source=1,
    random_seed=RANDOM_SEED,
):
    whole_df = load_whole_data_from_file(fmt="pkl")
    web_df = load_web_data_from_file(fmt="pkl")

    safe_conv_ids = set() # add your safety filterer: _safe_annotation_conv_ids()
    whole_df = _filter_to_safe_conversations(whole_df, safe_conv_ids)
    web_df = _filter_to_safe_conversations(web_df, safe_conv_ids)

    prior_turns = _history_turn_depth(history_depth)
    if prior_turns is None:
        prior_turns = 1

    def _starter_filter(df):
        df = df.copy()
        df["user_msg_history"] = df["user_msg_history"].apply(_as_list)
        df["assistant_msg_history"] = df["assistant_msg_history"].apply(_as_list)
        history_mask = df.apply(
            lambda row: _has_exact_history_depth(row, prior_turns),
            axis=1,
        )
        return df[history_mask & (df["language"] == "en")].copy()

    web_filtered = _starter_filter(web_df)
    web_keys = set(zip(web_filtered["conv_id"], web_filtered["turn_id"]))

    whole_filtered = _starter_filter(whole_df)
    whole_non_web = whole_filtered[
        ~whole_filtered.apply(
            lambda row: (row["conv_id"], row["turn_id"]) in web_keys, axis=1
        )
    ].copy()

    # print(len(web_filtered))
    # print(len(whole_non_web))

    def _sample(df):
        if samples_per_source is None:
            return df.copy()
        return df.sample(
            min(samples_per_source, len(df)),
            random_state=random_seed,
        ).copy()

    web_sample = _sample(web_filtered)
    web_sample["sample_source"] = "web"

    whole_sample = _sample(whole_non_web)
    whole_sample["sample_source"] = "non_web"

    df = (
        pd.concat([web_sample, whole_sample], ignore_index=True)
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )
    return df[
        df.apply(lambda row: _has_exact_history_depth(row, prior_turns), axis=1)
    ].copy()


tool_choices = ["auto", "none", "required"]


def _most_frequent_model(openai_models):
    valid_models = [
        m
        for m in openai_models
        if isinstance(m, str) and m.strip() and m.lower() != "none"
    ]
    if not valid_models:
        return None
    return Counter(valid_models).most_common(1)[0][0]


def _save_replay_results(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def _build_prompt(user_prompts, assistant_prompts, with_history, history_depth):
    user_prompts = _clean_messages(user_prompts)
    assistant_prompts = _clean_messages(assistant_prompts)
    if not user_prompts:
        raise ValueError("row has no user messages to replay")

    current_user_idx = len(user_prompts) - 1
    current_user_prompt = user_prompts[current_user_idx]
    prompt = []

    if with_history:
        paired_history = list(zip(user_prompts, assistant_prompts))[:current_user_idx]
        max_prior_turns = len(paired_history)
        history_turns = _history_turn_depth(history_depth)
        if history_turns is None:
            history_turns = max_prior_turns
        history_turns = min(history_turns, max_prior_turns)

        selected_history = paired_history[-history_turns:] if history_turns else []
        for user_prompt, assistant_prompt in selected_history:
            prompt.append({"role": "user", "content": user_prompt})
            prompt.append({"role": "assistant", "content": assistant_prompt})

    prompt.append({"role": "user", "content": current_user_prompt})
    return prompt, current_user_prompt


def _replay_result_key(row):
    return "::".join(
        str(row[column])
        for column in ["sample_source", "conv_id", "turn_id"]
    )


def replayer(
    model,
    with_history=False,
    save_every=5,
    output_file=None,
    history_depth=4,
    samples_per_source=1,
    random_seed=RANDOM_SEED,
):
    df = filter_df_for_history(
        history_depth,
        samples_per_source=samples_per_source,
        random_seed=random_seed,
    )
    model_results = {}
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df)), start=1):
        prompt, user_prompt = _build_prompt(
            row["user_msg_history"],
            row["assistant_msg_history"],
            with_history=with_history,
            history_depth=history_depth,
        )
        result_key = _replay_result_key(row)
        duplicate_idx = 2
        while result_key in model_results:
            result_key = f"{_replay_result_key(row)}::{duplicate_idx}"
            duplicate_idx += 1

        invivo_response = " ".join(
            json.loads(row["turn_msgs"])[-1].get("content", {}).get("parts", [])
        ).strip()
        row_model = _most_frequent_model(row["openai_models"])
        row_model = model_replacements.get(row_model, row_model)

        model_results[result_key] = {
            "result_key": result_key,
            "user_prompt": user_prompt,
            "prompt": prompt,
            "sample_source": row["sample_source"],
            "conv_id": row["conv_id"],
            "turn_id": row["turn_id"],
            "invivo_model": row_model,
            "replay_model": model,
            "invivo_response": invivo_response,
        }

        replay_model = row_model if model == "invivo" else model

        for tool_choice in tool_choices:
            # print(tool_choice)
            try:
                kwargs = {
                    "model": replay_model,
                    "input": prompt,
                    "include": ["web_search_call.action.sources"],
                    "store": False,
                    "tool_choice": tool_choice,
                    "tools": [{"type": "web_search"}]
                }

                if tool_choice == "none":
                    kwargs["tools"] = []

                response = client.responses.create(**kwargs)
                payload = {
                    "output_text": response.output_text,
                    "response": response.model_dump(),
                }
            except Exception:
                payload = {
                    "output_text": "",
                    "response": {},
                }
            model_results[result_key][tool_choice] = payload

        if output_file and save_every and idx % save_every == 0:
            _save_replay_results(model_results, output_file)

    if output_file:
        _save_replay_results(model_results, output_file)
    return model_results


if __name__ == "__main__":
    models = ["gpt-5-mini-2025-08-07"] # ["invivo"]

    for model in models:
        print(model)
        output_file = f"{OUTPUT_PATH}/replays/{model}.json"
        model_results = replayer(
            model,
            with_history=False,
            history_depth=0,
            save_every=5,
            output_file=output_file,
            samples_per_source=500
        )
        print(len(model_results))
