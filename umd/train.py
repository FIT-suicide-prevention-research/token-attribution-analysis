import argparse
import os
import shutil

import warnings

warnings.filterwarnings("ignore")

MODEL_LIST = ["mental/mental-roberta-base", "mental/mental-bert-base-uncased", "AIMH/mental-roberta-large", "AIMH/mental-bert-base-cased", "AIMH/mental-bert-large-uncased"]


def tokenize_df(df_corpus, model_name, seed=42):
    """
    Tokenize the corpus using the model_name.
    :param df_corpus: a pandas dataframe with two columns: text and label
    :param model_name: the name of the model
    :return: a dictionary with keys "input_ids", "attention_mask", "label"
    """
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load df into Dataset
    dataset = Dataset.from_pandas(df_corpus)
    # tokenize the corpus
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

    # format object to be used by the model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
    return tokenized_dataset

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("TRAIN_PATH", type=str, help="Path of training data.")
    parser.add_argument("TEST_PATH", type=str, help="Path of testing/validation data.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mental/mental-bert-base-uncased",
        help="The model to use for tokenization and fine-tuning. (default: mental/mental-bert-base-uncased)",
    )
    # add arg for batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to use for training and evaluation. (default: 32)",
    )

    args = parser.parse_args()
    return args

def train_trainer_ddp(args):

    # load data
    df_train = pd.read_parquet(args.TRAIN_PATH)
    df_test = pd.read_parquet(args.TEST_PATH)

    # tokenize the corpus
    tokenized_dataset_train = tokenize_df(df_train, args.model_name)
    tokenized_dataset_test = tokenize_df(df_test, args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=4)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # fine-tune the model
    training_args = TrainingArguments(
        output_dir=f"./results/{args.model_name}",          # output directory
        num_train_epochs=5,              # total # of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=f"./logs/{args.model_name}",            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        torch_compile=False,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=tokenized_dataset_train,         # training dataset
        eval_dataset=tokenized_dataset_test,            # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

    trainer.train()
    trainer.save_model(f"./models/{args.model_name}")
    shutil.rmtree(f"./results/{args.model_name}")

if __name__ == "__main__":
    args = args_parser()
    
    from transformers import logging

    logging.set_verbosity_warning()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    import pandas as pd
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    import evaluate
    from accelerate import notebook_launcher



    # train the model
    # notebook_launcher(train_trainer_ddp, [args], num_processes=1)
    train_trainer_ddp(args)