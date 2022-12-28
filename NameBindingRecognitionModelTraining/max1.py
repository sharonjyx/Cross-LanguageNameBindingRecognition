import torch
from transformers import BertTokenizer
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_csv(input_file,columns):
    with open(input_file,"r",encoding="utf-8") as file:
        lines=[]
        for line in file:
            if len(line.strip().split(",")) != 1:
                lines.append(line.strip().split(","))
        df = pd.DataFrame(lines)
        df.columns = columns
    return df
for p in ['itracker', 'sagan', 'springside', 'Tudu-Lists', 'zksample2', 'jrecruiter', 'hispacta', 'powerstone','jtrac', 'mall']:
    os.chdir(r"E:\nuaa\1st_Year\1_code\LocalGit\bert\\"+p)
    for i in ['match.csv']:#'uniquematch1.csv', 'newrepeatmatch.csv', 'newrepeatmatch2.csv'
        train = read_csv(i, ['text2', 'label'])
        train1 = train.loc[train['label'] == str(1), ['text2', 'label']]
        train0 = train.loc[train['label'] == str(0), ['text2', 'label']]
        text2 = train1.text2.values
        labels = train1.label.values
        labels = [int(num) for num in labels]

        text20 = train0.text2.values
        labels0 = train0.label.values
        labels0 = [int(num) for num in labels0]

        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        input_ids = []
        # token_type_ids = []
        attention_masks = []
        input_ids0 = []
        # token_type_ids0 = []
        attention_masks0 = []
        max_len = 0
        max_len0 = 0
        for i, row in train1.iterrows():
            input_ids = tokenizer.encode(row['text2'], add_special_tokens=True)
            # input_ids = tokenizer.encode(row['text1'], row['text2'], add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        print('Max sentence length: ', max_len)
        for i, row in train0.iterrows():
            input_ids0 = tokenizer.encode(row['text2'], add_special_tokens=True)
            # input_ids0 = tokenizer.encode(row['text1'], row['text2'], add_special_tokens=True)
            max_len0 = max(max_len0, len(input_ids0))
        print('Max sentence length: ', max_len0)