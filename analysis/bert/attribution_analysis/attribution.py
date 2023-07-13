from transformers_interpret import (
    SequenceClassificationExplainer,
    MultiLabelClassificationExplainer,
)
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json
torch.set_default_device("cuda")

import sys

DATA_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]
try:
    MODEL_NAME = sys.argv[3]
except IndexError:
    MODEL_NAME = "mental/mental-roberta-base"
# Load dataset
df_train = pd.read_parquet(f"{DATA_PATH}/train.parquet.gzip")
df_test = pd.read_parquet(f"{DATA_PATH}/test.parquet.gzip")
# Merge datasets
df = pd.concat([df_train, df_test]).reset_index(drop=True)

# Load the pre-trained and fine-tuned model
model_path = f"{MODEL_PATH}/{MODEL_NAME}"
model_name = MODEL_NAME
model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Make sure the model is in evaluation mode
model.eval()

# Add label
model.config.id2label = {
    0: "No Risk",
    1: "Low Risk",
    2: "Moderate Risk",
    3: "Severe Risk",
}
model.config.label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2, "LABEL_3": 3}


def gather_word_attributions(word_attributions_result, word_attributions_dict):
    for label, word_attributions in word_attributions_result.items():
        for i in range(len(word_attributions)):
            word_attributions_dict[word_attributions[i][0].lower()][label].append(
                word_attributions[i][1]
            )
    return word_attributions_dict


def plot_word_attributions(word_attributions_dict, word):
    bins = np.linspace(-0.5, 0.5, 100)
    plt.hist(word_attributions_dict[word]["LABEL_0"], bins, alpha=0.5, label="No Risk")
    plt.hist(word_attributions_dict[word]["LABEL_1"], bins, alpha=0.5, label="Low Risk")
    plt.hist(
        word_attributions_dict[word]["LABEL_2"], bins, alpha=0.5, label="Moderate Risk"
    )
    plt.hist(
        word_attributions_dict[word]["LABEL_3"], bins, alpha=0.5, label="Severe Risk"
    )
    plt.legend(loc="upper right")
    plt.show()


# Create explainer
cls_explainer = MultiLabelClassificationExplainer(model, tokenizer)

# Get word attributions
word_attributions_dict = defaultdict(lambda: defaultdict(list))
for text in df.text:
    # # truncate text through tokenizer
    # tokenized_text = tokenizer(text, truncation=True, max_length=450)
    # # convert tokenized text to text
    # text = tokenizer.decode(tokenized_text["input_ids"][1:-1])    
    word_attributions = cls_explainer(text)
    word_attributions_dict = gather_word_attributions(
        word_attributions, word_attributions_dict
    )

# Export word attributions
with open("word_attributions.json", "w") as f:
    json.dump(word_attributions_dict, f, indent=4)