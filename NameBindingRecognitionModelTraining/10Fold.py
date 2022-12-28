import pandas as pd
import os
from sklearn.model_selection import KFold
import numpy as np
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
    train=[]
    test=[]
    train1=[]
    train0=[]
    test1=[]
    test0=[]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(dataset1):
        dataset1 = np.array(dataset1)
        train_dataset =pd.DataFrame(dataset1[train_index.tolist()])
        test_dataset = pd.DataFrame(dataset1[test_index.tolist()])
        train1.append(train_dataset)
        test1.append(test_dataset)
    for train_index, test_index in kf.split(dataset0):
        dataset0 = np.array(dataset0)
        train_dataset =pd.DataFrame(dataset0[train_index.tolist()])
        test_dataset = pd.DataFrame(dataset0[test_index.tolist()])
        train0.append(train_dataset)
        test0.append(test_dataset)
    for train_dataset0 , train_dataset1,val_dataset0 , val_dataset1 in zip(train0,train1,test0,test1):
        train_dataset = pd.concat([train_dataset0 , train_dataset1], axis=0, ignore_index=True)
        val_dataset = pd.concat([val_dataset0 , val_dataset1], axis=0, ignore_index=True)
        train.append(train_dataset)
        test.append(val_dataset)
    return train,test
def write(train_dataset, val_dataset,index):
    train_dataset.to_csv(output_dir+'train'+str(index)+'.csv', index=False, header=False)
    val_dataset.to_csv(output_dir + 'test' +str(index)+ '.csv', index=False, header=False)
for project in ['springside']:#'itracker', 'sagan', 'springside', 'Tudu-Lists', 'zksample2','jrecruiter','hispacta','powerstone','jtrac','mall'
    print(project)
    for filename in ['uniquematchkv1.csv','newrepeatmatchni.csv']:
        train, val = get_dataset(project, filename)
        output_dir = './' + filename.replace('.csv', '') + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for train_dataset, val_dataset,index in zip(train, val,[1,2,3,4,5,6,7,8,9,10]):
            write(train_dataset, val_dataset,index)