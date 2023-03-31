import argparse
import os

import warnings

warnings.filterwarnings("ignore")


def _corpus_loader(df_post_path, df_crowd_path, task="A", include_title=False):
    """
    Load the corpus from the csv files.
    :param df_post_path: path to the csv file containing raw posts
    :param df_crowd_path: path to the csv file containing crowd labels
    :param task: task to load, either "A", "B" or "C"
        - **Task A**: Risk Assessment for SuicideWatch posters based *only* on their
        SuicideWatch postings.
        - **Task B**: Risk Assessment for SuicideWatch posters based on their
        SuicideWatch postings *and* other Reddit postings.
        - **Task C**: Screening. This task looks at posts that are *NOT* on
        SuicideWatch, and determine the user's level of risk.
    :param include_title: whether to include the post title in the corpus
    :return: a dataframe with two columns: text and label
        'd': '(d) Severe Risk',
        'c': '(c) Moderate Risk',
        'b': '(b) Low Risk',
        'a': '(a) No Risk'
    """
    df_post = pd.read_csv(df_post_path)
    df_crowd = pd.read_csv(df_crowd_path)
    df_suicidewatch = pd.merge(
        df_post[df_post["subreddit"] == "SuicideWatch"], df_crowd, on="user_id"
    )
    # create a dataframe with all post body and label
    df_suicidewatch_body = df_suicidewatch[["post_body", "label"]].copy()
    # rename the column name
    df_suicidewatch_body = df_suicidewatch_body.rename(columns={"post_body": "text"})

    if include_title:
        # create a dataframe with all post title and label
        df_suicidewatch_title = df_suicidewatch[["post_title", "label"]]
        df_suicidewatch_title.rename(columns={"post_title": "text"}, inplace=True)

        # concat the two dataframe
        df_suicidewatch_corpus = pd.concat(
            [df_suicidewatch_title, df_suicidewatch_body], ignore_index=True
        )
    else:
        df_suicidewatch_corpus = df_suicidewatch_body
    # remove rows with text length less than 2
    df_suicidewatch_corpus = df_suicidewatch_corpus[
        df_suicidewatch_corpus["text"].str.split().str.len() > 1
    ]

    return df_suicidewatch_corpus


# slice text into batch of sliding windows of 512 tokens stride 256, form into list of strings, apply to row text
def _sliding_window(text, window_size=512, stride=256):
    text = text.split()
    text = [text[i : i + window_size] for i in range(0, len(text), stride)]
    # remove last window if it is less than window_size
    if len(text) > 1:
        if len(text[-2]) < window_size:
            text = text[:-1]
    text = [" ".join(t) for t in text]
    return text


def _sliding_window_corpus(df_corpus, window_size=512, stride=256):
    """
    Slice the text in the corpus into sliding windows with stride.
    :param df_corpus: a dataframe with two columns: text and label
    :param window_size: the size of the sliding window
    :param stride: the stride of the sliding window
    :return: a dataframe with two columns: text and label
    """
    # apply sliding window to each text
    df_corpus["text"] = df_corpus["text"].apply(
        _sliding_window, args=(window_size, stride)
    )
    # explode the sliding windows into rows
    df_corpus = df_corpus.explode("text")
    return df_corpus


def preprocess(
    df_post_path,
    df_crowd_path,
    task="A",
    include_title=False,
    window_size=512,
    stride=256,
):
    """
    Preprocess the corpus from the csv files.
    :params df_post_path, df_crowd_path: path to the csv files
    :params task, include_title: see _corpus_loader()
    :params window_size, stride: see _sliding_window_corpus()
    :return: a dataframe with two columns: text and label
        Label is re-encoded as:
        0: '(a) No Risk'
        1: '(b) Low Risk'
        2: '(c) Moderate Risk'
        3: '(d) High Risk'
    """
    # load corpus
    df_corpus = _corpus_loader(
        df_post_path, df_crowd_path, task=task, include_title=include_title
    )

    # apply sliding window to each text
    df_corpus = _sliding_window_corpus(
        df_corpus, window_size=window_size, stride=stride
    )

    # reset index
    df_corpus.reset_index(drop=True, inplace=True)
    # encode label a b c d to 0 1 2 3
    df_corpus["label"] = df_corpus["label"].map({"a": 0, "b": 1, "c": 2, "d": 3})

    return df_corpus


# def tokenize_df(df_corpus, model_name, seed=42):
#     """
#     Tokenize the corpus using the model_name.
#     :param df_corpus: a pandas dataframe with two columns: text and label
#     :param model_name: the name of the model
#     :return: a dictionary with keys "input_ids", "attention_mask", "label"
#     """
#     # load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # load df into Dataset
#     dataset = Dataset.from_pandas(df_corpus)
#     # tokenize the corpus
#     tokenized_dataset = dataset.map(
#         lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
#         batched=True,
#     )

#     # format object to be used by the model
#     tokenized_dataset = tokenized_dataset.remove_columns(["text"])
#     tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
#     tokenized_dataset.set_format("torch")
#     tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
#     return tokenized_dataset


def args_parser():
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "POST_PATH",
        type=str,
        help="The path to the csv file containing the post data.",
    )
    parser.add_argument(
        "LABEL_PATH",
        type=str,
        help="""
        The path to the csv file containing the label data.
        For training, use crowd_train.csv.
        For testing/validation, use crowd_test_{task}.csv.
        """,
    )
    parser.add_argument(
        "OUTPUT_FILE",
        type=str,
        help="The path to save the tokenized corpus data.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="A",
        help="The task to perform. Options: A, B, C. (default: A)",
    )
    parser.add_argument(
        "--include_title",
        type=bool,
        default=False,
        help="Whether to include the post title in the corpus. (default: False)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=512,
        help="The size of the sliding window. (default: 512)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="The stride of the sliding window. (default: 256))",
    )
    return parser


if __name__ == "__main__":
    # parse arguments
    parser = args_parser()
    args = parser.parse_args()

    import pandas as pd

    # load corpus
    df_corpus = preprocess(
        args.POST_PATH,
        args.LABEL_PATH,
        task=args.task,
        include_title=args.include_title,
        window_size=args.window_size,
        stride=args.stride,
    )

    # save the corpus
    os.makedirs(os.path.dirname(args.OUTPUT_FILE), exist_ok=True)
    df_corpus.to_parquet(args.OUTPUT_FILE, compression="gzip")
    #     # tokenize the corpus
    # tokenized_dataset = tokenize_df(df_corpus, args.model_name, seed=args.seed)

    # # save the tokenized corpus
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # torch.save(tokenized_dataset, args.output_file)
