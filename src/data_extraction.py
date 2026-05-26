import ast
import sys

sys.setrecursionlimit(5000)
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import pandas as pd
from data_utils import *
from urllib.parse import urlparse
try:
    from langdetect import detect
except ImportError:
    def detect(_text):
        return ""

NUM_USERS = 310


def load_whole_data():
    all_data = []
    num_conversations = 0
    num_turns = 0
    num_msgs = 0
    num_tool_usage = 0
    tools = []
    topic_lookup = load_topics()

    # Load all files
    for i in tqdm(range(NUM_USERS)):
        file_path = Path(
            f"{DATA_BASE_PATH}/prolific_all_files/user_{i}/conversations.json"
        )
        if not file_path.exists():
            continue

        with open(file_path, "r") as f:
            data = json.load(f)

        for conv_idx, conv in enumerate(data):
            num_conversations += 1
            mapping = sort_conversation(conv["mapping"])
            user_msg_history = []
            assistant_msg_history = []

            turn_msgs = []
            conv_starter = 1
            for msg_info in mapping:
                msg = msg_info.get("message")
                if not msg:
                    continue

                role = msg.get("author", {}).get("role", "")
                if role == "system":
                    continue

                turn_msgs.append(msg)
                if role == "user":
                    temp = msg.get("content", {}).get("parts", [])
                    user_msg_history += [str(x) for x in temp]

                if role == "assistant" and msg.get("recipient", "") == "all":
                    temp = msg.get("content", {}).get("parts", [])
                    assistant_msg_history += [str(x) for x in temp]

                end_turn = msg.get("end_turn")

                if end_turn:
                    num_turns += 1
                    # eval turn msgs
                    main_tool_calls = []
                    reasoning_path = []
                    thinking_path = []
                    openai_models = []
                    interactions = []
                    user_query = []
                    thoughts = ""
                    for turn_msg in turn_msgs:
                        role = turn_msg.get("author", {}).get("name", "")
                        if not role:
                            role = turn_msg.get("author", {}).get("role", "")

                        if not user_query and role == "user":
                            user_query = turn_msg["content"].get("parts", [])

                        recipient = turn_msg.get("recipient")
                        ts = turn_msg.get("create_time")
                        openai_models.append(
                            turn_msg["metadata"].get("model_slug", None)
                        )
                        reasoning_path.append(
                            turn_msg["metadata"].get("reasoning_status", None)
                        )
                        thinking_type = turn_msg["content"].get("content_type", None)
                        thinking_thoughts = turn_msg["content"].get("thoughts", [])
                        thinking_path.append(thinking_type)
                        if "thoughts" == thinking_type:
                            for tt in thinking_thoughts:
                                thoughts += tt.get("content", "")
                                thoughts += "\n"

                        interactions.append(f"{role}:{recipient}")
                        if ts and role == "assistant" and recipient != "all":
                            main_tool_calls.append(recipient)
                            num_tool_usage += 1

                    reasoning = False
                    for path in reasoning_path:
                        if path:
                            reasoning = True
                            break

                    thinking = "thoughts" in thinking_path

                    time_ = normalize_timestamp(turn_msgs[-1].get("create_time"))
                    topic_ = str(topic_lookup.get(conv["id"], "Other"))
                    topic = topic_ if topic_ != "nan" else "Other"
                    language = ""
                    try:
                        language = detect("\n".join([str(x) for x in user_msg_history]))
                    except:
                        language = ""
                    all_data.append(
                        [
                            i,
                            conv["id"],
                            turn_msgs[0]["id"],
                            conv_starter,
                            topic,
                            language,
                            main_tool_calls,
                            interactions,
                            int(reasoning),
                            int(thinking),
                            thoughts,
                            openai_models,
                            user_msg_history.copy(),
                            assistant_msg_history.copy(),
                            json.dumps(turn_msgs),
                            time_,
                        ]
                    )
                    tools += main_tool_calls
                    conv_starter = 0

                    # empty turn msgs for new one
                    num_msgs += len(turn_msgs)
                    turn_msgs = []

    return all_data, num_conversations, num_turns, num_msgs, num_tool_usage


def load_whole_data_from_file(fmt, base_dir=None):
    metadata_dir = Path(base_dir) if base_dir else Path(f"{OUTPUT_PATH}/metadata")
    if fmt == "csv":
        df = pd.read_csv(metadata_dir / "data_summary.csv")
    elif fmt == "pkl":
        df = pd.read_pickle(metadata_dir / "data_summary.pkl")
    elif fmt == "parquet":
        df = pd.read_parquet(metadata_dir / "data_summary.parquet")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return df


def load_web_data_from_file(fmt, base_dir=None):
    metadata_dir = Path(base_dir) if base_dir else Path(f"{OUTPUT_PATH}/metadata")
    if fmt == "csv":
        df = pd.read_csv(metadata_dir / "web_data_summary.csv")
    elif fmt == "pkl":
        df = pd.read_pickle(metadata_dir / "web_data_summary.pkl")
    elif fmt == "parquet":
        df = pd.read_parquet(metadata_dir / "web_data_summary.parquet")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return df


def main():
    all_data, num_conversations, num_turns, num_msgs, num_tool_usage = load_whole_data()
    # Convert to DataFrame
    df = pd.DataFrame(
        all_data,
        columns=[
            "user_id",
            "conv_id",
            "turn_id",
            "conv_starter",
            "topic",
            "language",
            "tools",
            "interactions",
            "reasoning",
            "thinking",
            "thoughts",
            "openai_models",
            "user_msg_history",
            "assistant_msg_history",
            "turn_msgs",
            "time",
        ],
    )
    df = df.sort_values("time")
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

    print(f"Number of Users: {NUM_USERS}")
    print(f"Number of Conversation: {num_conversations}")
    print(f"Number of Turns: {num_turns}")
    print(f"Number of Turns with Tool calls: {len(df[df['tools'].apply(len) > 0])}")
    print(f"Number of Messages: {num_msgs}")
    print(f"Number of Messages with Tool calls: {num_tool_usage}")

    df = df.reset_index(drop=True)
    df.to_parquet(f"{OUTPUT_PATH}/metadata/data_summary.parquet")
    df.to_pickle(f"{OUTPUT_PATH}/metadata/data_summary.pkl")
    df.to_csv(f"{OUTPUT_PATH}/metadata/data_summary.csv", index=False)
    print("All Data Saved Successfully!")

    web_df = df[(df["interactions"].apply(lambda x: "web" in str(x)))]

    web_df = web_df.reset_index(drop=True)
    web_df.to_parquet(f"{OUTPUT_PATH}/metadata/web_data_summary.parquet")
    web_df.to_pickle(f"{OUTPUT_PATH}/metadata/web_data_summary.pkl")
    web_df.to_csv(f"{OUTPUT_PATH}/metadata/web_data_summary.csv", index=False)
    print("Web Data Saved Successfully!")


if __name__ == "__main__":
    main()


# # OUTPUT:
# Number of Users: 310
# Number of Conversation: 143730
# Number of Turns: 681426
# Number of Turns with Tool calls: 73708
# Number of Messages: 1781927
# Number of Messages with Tool calls: 97072
