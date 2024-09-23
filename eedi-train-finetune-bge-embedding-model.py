import os, re, json
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils import run_script

tqdm.pandas()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_id = 'BAAI/bge-large-en-v1.5'
comp_dir = './data/'


train = pd.read_csv(f'{comp_dir}/train.csv')
misconceptions = pd.read_csv(f'{comp_dir}/misconception_mapping.csv')

train["AllQuestionText"] = train["SubjectName"] + "\n\n" + train["ConstructName"] + "\n\n" + train["QuestionText"]

keep_cols = ["QuestionId", "AllQuestionText", "CorrectAnswer"]
answer_cols = ["AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText"]
misconception_cols = ["MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId"]


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    # Melt the answer columns
    answers_df = pd.melt(
        id_vars=keep_cols,
        frame=df[keep_cols + answer_cols],
        var_name='Answer', value_name='Value'
    ).sort_values(["QuestionId", "Answer"]).reset_index(drop=True)

    # If NOT test set
    if misconception_cols[0] in df.columns:
        # Melt the misconception columns
        misconceptions_df = pd.melt(
            id_vars=keep_cols,
            frame=df[keep_cols + misconception_cols],
            var_name='Misconception', value_name='MisconceptionId'
        ).sort_values(["QuestionId", "Misconception"]).reset_index(drop=True)

        answers_df[['Misconception', 'MisconceptionId']] = misconceptions_df[['Misconception', 'MisconceptionId']]

    return answers_df


train = wide_to_long(train)


def preprocess_text(x):
    x = x.lower()                 # Convert words to lowercase
    x = re.sub("@\w+", '',x)      # Delete strings starting with @
    x = re.sub("'\d+", '',x)      # Delete Numbers
    x = re.sub("\d+", '',x)
    x = re.sub("http\w+", '',x)   # Delete URL
    x = re.sub(r"\s+", " ", x)    # Replace consecutive empty spaces with a single space character
    x = re.sub(r"\.+", ".", x)    # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = x.strip()                 # Remove empty characters at the beginning and end
    return x



train["AllText"] = train["AllQuestionText"] + "\n\n" + train["Value"]
train['AnswerId'] = train.Answer.str.replace('Answer', '').str.replace('Text', '')

train = train[train.AnswerId != train.CorrectAnswer].reset_index(drop=True)
train.drop(['AllQuestionText', 'Answer', 'Misconception'], axis=1, inplace=True)

train = pd.merge(train, misconceptions, on='MisconceptionId', how='left')

train = train.dropna()

train["AllText"] = train["AllText"].apply(preprocess_text)
train["MisconceptionName"] = train["MisconceptionName"].apply(preprocess_text)

len(train)

train.head()

tokenizer = AutoTokenizer.from_pretrained(model_id)

all_texts_len = train['AllText'].progress_apply(lambda x: len(tokenizer(x)['input_ids']))
misconceptions_len = misconceptions['MisconceptionName'].progress_apply(lambda x: len(tokenizer(x)['input_ids']))

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
#
# _ = all_texts_len.sort_values().reset_index(drop=True).plot.line(ax=ax1)
# _ = misconceptions_len.sort_values().reset_index(drop=True).plot.line(ax=ax2)



pretrain_data = [{'text': preprocess_text(misconception)} for misconception in list(misconceptions.MisconceptionName.values)]

finetune_data = [
    {
        'query': query.strip(),
        'pos': [misconception.strip()],
        'neg': []    # Leave empty, to be populated by hard mining algorithm below
    } for query, misconception in train[['AllText', 'MisconceptionName']].values
]

print(len(pretrain_data), len(finetune_data))

print(finetune_data[0])

with open('pretrain_data.jsonl', 'w') as f:
    for entry in pretrain_data:
        json.dump(entry, f)
        f.write('\n')

with open('finetune_data.jsonl', 'w') as f:
    for entry in finetune_data:
        json.dump(entry, f)
        f.write('\n')



script_content = f"""
cd /home/seazer/code/MathLLM
/home/seazer/.pyenv/versions/3.11.9/envs/MathLLM/bin/python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine --model_name_or_path BAAI/bge-large-en-v1.5 --input_file finetune_data.jsonl --candidate_pool pretrain_data.jsonl --output_file finetune_data_minedHN.jsonl --range_for_sampling 1-100 --negative_number 15 --use_gpu_for_searching
"""

run_script(script_content)

finetune_data_minedHN = []
with open('finetune_data_minedHN.jsonl', 'r') as file:
    for line in file:
        finetune_data_minedHN.append(json.loads(line))

print(finetune_data_minedHN[0])

script_content = f"""
torchrun --nproc_per_node 2 -m FlagEmbedding.baai_general_embedding.finetune.run --output_dir eedi_model --model_name_or_path BAAI/bge-large-en-v1.5 --train_data finetune_data_minedHN.jsonl --learning_rate 1e-5 --fp16 --temperature 0.03 --num_train_epochs 3 --per_device_train_batch_size 8 --query_max_len 256 --passage_max_len 64 --logging_steps 100 --query_instruction_for_retrieval "" --report_to none --save_steps 250
"""

run_script(script_content)


if __name__ == '__main__':
    print("Done")