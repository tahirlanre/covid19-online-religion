# code adapted from https://github.com/umanlp/RedditBias/blob/61f9ae9458e26a64ae1113425192280503f8fb92/Evaluation/measure_bias.py

import math
import json
import logging

import pandas as pd
import numpy as np
from scipy import stats
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.utils import init_logger

logger = logging.getLogger(__name__)
init_logger()


def get_perplexity_score(text, model, tokenizer):
    with torch.no_grad():
        model.eval()
        tokenize_input = tokenizer.tokenize(text)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = model(tensor_input, labels=tensor_input)
        return math.exp(loss[0])


def get_model_perplexity(text, model, tokenizer):
    pass


def get_perplexity_list(df, model, tokenizer):
    """
    Gets perplexities of all sentences in a DataFrame based on given model
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model
    Returns
    -------
    List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            perplexity = get_perplexity_score(row["text"], model, tokenizer)
        except Exception as ex:
            print(ex.__repr__())
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def find_anomalies(data):
    """
    Find outliers in a given data distribution
    Parameters
    ----------
    data : list
    List of sentence perplexities
    Returns
    -------
    List of outliers
    """
    anomalies = []

    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies


output_dir = "../saved_output/language_modeling"
lm_mapping = json.load(open("../data/lm_mapping.json"))
data_path = "../data/twitter/"
mode = "online"  # online or offline
activity = "worship"

GET_PERPLEXITY = True

# month = "07" # 07, 08 or 09
# year = "19" # 19 or 20
month = "07"
period1 = f"{month}_19"
period2 = f"{month}_20"

df = pd.read_csv(data_path + "twitter_questions_" + mode + "_" + activity + ".csv")

if GET_PERPLEXITY:
    logging.info(f"Calculating perplexity for activity: {activity}")

    model_path = f"{output_dir}/{lm_mapping[f'{period1}']}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    period1_perplexity = get_perplexity_list(df, model, tokenizer)

    model_path = f"{output_dir}/{lm_mapping[f'{period2}']}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    period2_perplexity = get_perplexity_list(df, model, tokenizer)

    df[period1] = period1_perplexity
    df[period2] = period2_perplexity

    df.to_csv(
        data_path + "twitter_questions_" + mode + "_" + activity + ".csv", index=False
    )

else:
    logging.info(f"Getting save perplexity for activity: {activity}")
    period1_perplexity = df[period1]
    period2_perplexity = df[period2]

logging.debug(
    "Mean and variance of unfiltered perplexities period1 - Mean {}, Variance {}".format(
        np.mean(period1_perplexity), np.var(period1_perplexity)
    )
)
logging.debug(
    "Mean and variance of unfiltered perplexities period2 - Mean {}, Variance {}".format(
        np.mean(period2_perplexity), np.var(period2_perplexity)
    )
)

print(
    "Mean and std of unfiltered perplexities period1 - Mean {}, Std {}".format(
        np.mean(period1_perplexity), np.std(period1_perplexity)
    )
)
print(
    "Mean and std of unfiltered perplexities period2 - Mean {}, Std {}".format(
        np.mean(period2_perplexity), np.std(period2_perplexity)
    )
)

period1_out = find_anomalies(np.array(period1_perplexity))
period2_out = find_anomalies(np.array(period2_perplexity))

print(period1_out, period2_out)
demo1_in = [d1 for d1 in period1_perplexity if d1 not in period1_out]
demo2_in = [d2 for d2 in period2_perplexity if d2 not in period2_out]

for i, (p1, p2) in enumerate(zip(period1_perplexity, period2_perplexity)):
    if p1 in period1_out or p2 in period2_out:
        print("Outlier in demo1 is {}".format(df.loc[df[period1] == p1]))
        print("Outlier in demo2 is {}".format(df.loc[df[period1] == p2]))
        df.drop(df.loc[df[period1] == p1].index, inplace=True)
        df.drop(df.loc[df[period2] == p2].index, inplace=True)

t_value, p_value = stats.ttest_rel(period1_perplexity, period2_perplexity)

print(
    "Mean and std of unfiltered perplexities demo1 - Mean {}, Std {}".format(
        np.mean(period1_perplexity), np.std(period2_perplexity)
    )
)
print(
    "Mean and std of unfiltered perplexities demo2 - Mean {}, Std {}".format(
        np.mean(period2_perplexity), np.std(period2_perplexity)
    )
)
print("Unfiltered perplexities - T value {} and P value {}".format(t_value, p_value))
print(t_value, p_value)

confidence_level = 0.05
if p_value < confidence_level:
    print("it is significant")
else:
    print("it is not significant")
