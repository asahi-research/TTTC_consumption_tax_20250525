"""Create summaries for the clusters."""

from tqdm import tqdm
import os
from typing import List
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from utils import messages, update_progress
import openai

def overview(config):
    dataset = config['output_dir']
    path = f"outputs/{dataset}/overview.txt"

    takeaways = pd.read_csv(f"outputs/{dataset}/takeaways.csv")
    labels = pd.read_csv(f"outputs/{dataset}/labels.csv")

    prompt = config['overview']['prompt']
    model = config['overview']['model']

    ids = labels['cluster-id'].to_list()
    takeaways.set_index('cluster-id', inplace=True)
    labels.set_index('cluster-id', inplace=True)

    input = ''
    for i, id in enumerate(ids):
        input += f"# Cluster {i}/{len(ids)}: {labels.loc[id]['label']}\n\n"
        input += takeaways.loc[id]['takeaways'] + '\n\n'

    #llm = ChatOpenAI(model_name=model, temperature=0.0)
    #response = llm(messages=messages(prompt, input)).content.strip()
    # OpenAIクライアント初期化
    llm = openai.OpenAI()
    # モデル呼び出し
    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input},
        ],
        temperature=1
    ).choices[0].message.content.strip()
    with open(path, 'w') as file:
        file.write(response)
