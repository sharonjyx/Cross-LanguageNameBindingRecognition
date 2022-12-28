import argparse
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer,BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import random
import time
import datetime
import os
from sklearn.metrics import confusion_matrix
from focal_loss import FocalLoss
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
def read_csv(input_file,columns):
    with open(input_file,"r",encoding="utf-8") as file:
        lines=[]
        for line in file:
            if len(line.strip().split(",")) != 1:
                lines.append(line.strip().split(","))
        df = pd.DataFrame(lines)
        df.columns = columns
    return df
def get_dataset(project,filename):
    os.chdir(r"E:\nuaa\1st_Year\1_code\LocalGit\bert\\" + project)
    dataset = read_csv(filename, ['text2', 'label'])
    dataset1 = dataset.loc[dataset['label'] == str(1), ['text2', 'label']]
    dataset0 = dataset.loc[dataset['label'] == str(0), ['text2', 'label']]
    train_dataset1, val_dataset1 = train_test_split(dataset1, test_size=0.2, random_state=42)
    train_dataset0, val_dataset0 = train_test_split(dataset0, test_size=0.2, random_state=42)
    train_dataset = pd.concat([train_dataset0 , train_dataset1], axis=0, ignore_index=True)
    dataset1 = train_dataset.loc[train_dataset['label'] == str(1), ['text2', 'label']]
    train_dataset = pd.concat([train_dataset,dataset1,dataset1,dataset1,dataset1], axis=0, ignore_index=True)
    val_dataset = pd.concat([val_dataset0 , val_dataset1], axis=0, ignore_index=True)
    return train_dataset,val_dataset
def plot(df_stats):
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)
    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4, 5])
    plt.show()
def train(model, train_dataloader, optimizer, device, scheduler, epoch_i, loss_func):
    print("")
    print('======== Epoch {:} ========'.format(epoch_i + 1))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = 0
    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # model.zero_grad()
        optimizer.zero_grad()
        loss, logits = model(b_input_ids,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        logits = logits.softmax(dim=1)
        # loss=loss_func(logits,b_labels)
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    return training_time, avg_train_loss
def validation(model, validation_dataloader, device, epoch_i, training_stats, training_time, avg_train_loss, loss_func):
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables
    total_eval_loss = 0
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            loss, logits = model(b_input_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
        # Accumulate the validation loss.
        # loss = loss_func(logits, b_labels)
        total_eval_loss += loss.item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred = np.argmax(logits, axis=1)
        test = label_ids

        print(confusion_matrix(test, pred, labels=[0, 1]).ravel())
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(test, pred,
                                                                                        labels=[0, 1]).ravel()
        tn += true_negative
        fp += false_positive
        fn += false_negative
        tp += true_positive

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    if math.isnan(recall):
        recall = 0
    if math.isnan(precision):
        precision = 0
    if recall == 0 and precision == 0:
        F1 = 0
    else:
        F1 = (2 * precision * recall) / (precision + recall)
    if math.isnan(F1):
        F1 = 0
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    print("  recall: {}".format(recall))
    print("  precision: {}".format(precision))
    print("  F1: {}".format(F1))
    print("  Validation Loss: {}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. recall': recall,
            'Valid. precision': precision,
            'Valid. F1': F1,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    return model
def load_data(tokenizer, train_dataset, val_dataset, max_length, batch_size):
    # print(type(train_dataset))
    labels1 = train_dataset.label.values
    labels1 = [int(num) for num in labels1]
    labels0 = val_dataset.label.values
    labels0 = [int(num) for num in labels0]
    # Tokenize Dataset
    input_ids1 = []
    attention_masks1 = []
    for i, row in train_dataset.iterrows():
        encoded_dict = tokenizer.encode_plus(
            row['text2'],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids1.append(encoded_dict['input_ids'])
        attention_masks1.append(encoded_dict['attention_mask'])
    input_ids0 = []
    attention_masks0 = []
    for i, row in val_dataset.iterrows():
        encoded_dict = tokenizer.encode_plus(
            row['text2'],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids0.append(encoded_dict['input_ids'])
        attention_masks0.append(encoded_dict['attention_mask'])

    input_ids1 = torch.cat(input_ids1, dim=0)
    attention_masks1 = torch.cat(attention_masks1, dim=0)
    labels1 = torch.tensor(labels1)
    train_dataset = TensorDataset(input_ids1, attention_masks1, labels1)

    input_ids0 = torch.cat(input_ids0, dim=0)
    attention_masks0 = torch.cat(attention_masks0, dim=0)
    labels0 = torch.tensor(labels0)
    val_dataset = TensorDataset(input_ids0, attention_masks0, labels0)

    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
    )
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=RandomSampler(val_dataset),  # SequentialSampler Pull out batches sequentially.
        batch_size=batch_size,  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader
def main(project, epochs, learning_rate, batch_size, adam_epsilon, max_length, num_labels):
    os.chdir(r"E:\nuaa\1st_Year\1_code\LocalGit\bert\\" + project)
    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('Loading BERT tokenizer...')
    for filename in ['newrepeatmatch.csv']:#,'newrepeatmatch.csv'
        output_dir = './' + filename.replace('.csv', '') + '/less5data/'
        # output_dir = './model_save/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # model = BertForSequenceClassification.from_pretrained(
        #     "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        #     num_labels=num_labels,  # The number of output labels--2 for binary classification.
        #     # You can increase this for multi-class tasks.
        #     output_attentions=False,  # Whether the model returns attentions weights.
        #     output_hidden_states=False,  # Whether the model returns all hidden-states.
        # )
        model = Classify.from_pretrained(pretrained_model_name_or_path="bert-base-uncased", test=num_labels)
        model.to(device)
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        # Optimizer & Learning Rate Scheduler
        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=adam_epsilon  # args.adam_epsilon  - default is 1e-8.
                          )
        # loss_func=FocalLoss()
        loss_func = nn.CrossEntropyLoss()

        # Training Loop
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        training_stats = []
        train_dataset, val_dataset = get_dataset(project, filename)

        train_dataloader, validation_dataloader = load_data(tokenizer, train_dataset, val_dataset, max_length,
                                                            batch_size)
        total_steps = len(train_dataloader) * epochs
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        # checkpoint = torch.load('checkpoint.pth.tar')
        # epochs = 5-checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'])  # 加载模型的参数
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        print(epochs)
        for epoch_i in range(0, epochs):
            total_t = time.time()
            training_time, avg_train_loss = train(model, train_dataloader, optimizer, device, scheduler, epoch_i,
                                                  loss_func)
            model = validation(model, validation_dataloader, device, epoch_i, training_stats, training_time,
                               avg_train_loss, loss_func)
            state = {
                'epoch': epoch_i + 1,  # 保存当前的迭代次数
                'state_dict': model.state_dict(),  # 保存模型参数
                'optimizer': optimizer.state_dict(),  # 保存优化器参数
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, output_dir + 'checkpoint.pth.tar')
        df_stats = pd.DataFrame(data=training_stats)
        recall = 0
        precision = 0
        F1 = 0
        loss = 0
        validation_time = 0
        for index, row in df_stats.iterrows():
            recall += row['Valid. recall']
            precision += row['Valid. precision']
            F1 += row['Valid. F1']
            loss += row['Valid. Loss']
            validation_time = row['Validation Time']
        recall = recall / epochs
        precision = precision / epochs
        F1 = F1 / epochs
        loss = loss / epochs
        file = open('write1.csv', mode='a+', newline='')
        file.write(str(filename) + ',avg5,' + str("{}".format(
            recall)) + ',' + str("{}".format(precision)) + ',' + str("{}".format(F1)) + ',' + str(
            "{}".format(loss)) + ',' + str(
            "{}".format(validation_time)) + ',' + str(
            "{}".format(format_time(time.time() - total_t))) + '\n')
        df_stats = df_stats.set_index('epoch')
        plot(df_stats)
        # 使用transformers预训练后进行保存
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
class Classify(BertForSequenceClassification):
    def __init__(self, bert_dir, **kwargs):
        super(Classify, self).__init__(bert_dir)
        num_labels = kwargs.pop("test", 3)
        if device == 'cuda':
            self.dense_1 = nn.Linear(768, 768).cuda()
            self.dense_2 = nn.Linear(768, num_labels).cuda()
            self.dropout = nn.Dropout(0.5).cuda()
            self.softmax = nn.Softmax().cuda()
        else:
            self.dense_1 = nn.Linear(768, 768)
            self.dense_2 = nn.Linear(768, num_labels)
            self.dropout = nn.Dropout(0.5)
            self.softmax = nn.Softmax()
def predict(project, filename, device, max_length, batch_size):
    os.chdir(r"E:\nuaa\1st_Year\1_code\LocalGit\bert\\" + project)
    # 预训练模型使用 `from_pretrained()` 重新加载
    for a in range(5):
        output_dir = './' + filename.replace('.csv', '') + '/'
        dataset = read_csv(output_dir + 'test' + str(a + 1) + '.csv', ['text2', 'label'])
        labels = dataset.label.values
        labels = [int(num) for num in labels]
        output_dir = './' + filename.replace('.csv', '') + '/' + str(5) + '/'
        print(output_dir)
        if not os.path.exists(output_dir):
            return False
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        input_ids = []
        attention_masks = []
        for i, row in dataset.iterrows():
            encoded_dict = tokenizer.encode_plus(
                row['text2'],  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_length,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        test_dataset = TensorDataset(input_ids, attention_masks, labels)
        prediction_dataloader = DataLoader(
            test_dataset,
            sampler=RandomSampler(test_dataset),
            batch_size=batch_size,
        )

        config = BertConfig.from_pretrained(output_dir + 'config.json', output_hidden_states=True)
        model = BertForSequenceClassification.from_pretrained(output_dir, config=config)
        print('Predicting')
        t0 = time.time()
        model.eval()
        # Predict
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)
            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Store predictions and true labels
            pred = np.argmax(logits, axis=1)
            test = label_ids
            true_negative, false_positive, false_negative, true_positive = confusion_matrix(test, pred,
                                                                                            labels=[0, 1]).ravel()
            tn += true_negative
            fp += false_positive
            fn += false_negative
            tp += true_positive

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        if math.isnan(recall):
            recall = 0
        if math.isnan(precision):
            precision = 0
        if recall == 0 and precision == 0:
            F1 = 0
        else:
            F1 = (2 * precision * recall) / (precision + recall)
        if math.isnan(F1):
            F1 = 0
        test_time = format_time(time.time() - t0)
        print("  recall: {}".format(recall))
        print("  precision: {}".format(precision))
        print("  F1: {}".format(F1))
        print("  Test took: {:}".format(test_time))
        # Record all statistics from this epoch.
        file = open('write2.csv', mode='a+', newline='')
        file.write(
            str(filename) + ',' + str(a + 1) + ',' + str("{}".format(
                recall)) + ',' + str("{}".format(precision)) + ',' + str("{}".format(
                F1)) + '\n')
    print('DONE.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--adam_epsilon',type=float, default='1e-8')
    parser.add_argument('--max_length', type=int,default=64)
    parser.add_argument('--num_labels', type=int,default='2')
    args = parser.parse_args()
    # 128'itracker','Tudu-Lists','jrecruiter','powerstone', 'mall'
    # 64'zksample2','hispacta','jtrac'
    # 512'sagan', 'springside'
    # for project in ['powerstone', 'itracker', 'mall','Tudu-Lists', 'jrecruiter']:
    #     main(project,
    #          args.epoch,
    #          args.learning_rate,
    #          args.batch_size,
    #          args.adam_epsilon,
    #          128,
    #          args.num_labels)
    # for project in ['sagan', 'springside']:
    #     main(project,
    #          args.epoch,
    #          args.learning_rate,
    #          args.batch_size,
    #          args.adam_epsilon,
    #          512,
    #          args.num_labels)
    # for project in ['hispacta', 'jtrac', 'zksample2']:
    #     main(project,
    #          args.epoch,
    #          args.learning_rate,
    #          args.batch_size,
    #          args.adam_epsilon,
    #          args.max_length,
    #          args.num_labels)
    #predict('hispacta', 'newrepeatmatch.csv',device, args.max_length, args.batch_size)