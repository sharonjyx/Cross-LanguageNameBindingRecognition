import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
def read_csv(input_file,columns):
    with open(input_file,"r",encoding="utf-8") as file:
        lines=[]
        for line in file:
            if len(line.strip().split(",")) != 1:
                lines.append(line.strip().split(","))
        df = pd.DataFrame(lines)
        df.columns = columns
    return df
# for i in ['itracker','sagan','springside','Tudu-Lists','zksample2','mall','jrecruiter','hispacta','powerstone','jtrac']:#
#     os.chdir("E:\\nuaa\\1st_Year\\1_code\\LocalGit\\bert\\"+i)
    # for filename in ['uniquematch.csv','repeatmatch.csv']:
    #     df = read_csv(filename, ['text1', 'text2', 'label'])
    #     train1 = df.loc[df['label'] == str(1), ['text1', 'text2', 'label']]
    #     train0 = df.loc[df['label'] == str(0), ['text1', 'text2', 'label']]
    #     i0 = []
    #     for index, row in train1.iterrows():
    #         text1 = row['text1']
    #         text2 = row['text2']
    #         for index0, row0 in train0.loc[train0['text2'] == text2].iterrows():
    #             if text1 == row0['text1']:
    #                 # print(index)
    #                 # print(index0)
    #                 # print('----------------')
    #                 i0.append(index0)
    #     print(filename)
    #     print(len(i0))
    #     print(set(i0))
    #     print(len(set(i0)))
    #     print(len(df))
    #     df.drop(list(set(i0)), inplace=True)
    #     print(len(df))
    #     df.to_csv(filename, index=False, header=False)
    # for filename in ['repeatmatchless.csv', 'repeattoken.csv']:
    #     df = read_csv(filename, ['field','text1', 'text2', 'label', 'javaclass', 'nonjavafile','nonjavafilename','javapath','nonjavapath'])
    #     train1 = df.loc[df['label'] == str(1), ['field','text1', 'text2', 'label', 'javaclass', 'nonjavafile','nonjavafilename','javapath','nonjavapath']]
    #     train0 = df.loc[df['label'] == str(0), ['field','text1', 'text2', 'label', 'javaclass', 'nonjavafile','nonjavafilename','javapath','nonjavapath']]
    #     i0 = []
    #     for index, row in train1.iterrows():
    #         text1 = row['text1']
    #         text2 = row['text2']
    #         for index0, row0 in train0.loc[train0['text2'] == text2].iterrows():
    #             if text1 == row0['text1']:
    #                 # print(index)
    #                 # print(index0)
    #                 # print('----------------')
    #                 i0.append(index0)
    #     print(filename)
    #     print(len(i0))
    #     print(set(i0))
    #     print(len(set(i0)))
    #     print(len(df))
    #     df.drop(list(set(i0)), inplace=True)
    #     print(len(df))
    #     df.to_csv(filename, index=False, header=False)
    #     if filename == 'repeatmatchless.csv':
    #         print('repeattoken.csv')
    #         df2 = read_csv('repeattoken.csv', ['field','text1', 'text2', 'label', 'javaclass', 'nonjavafile','nonjavafilename','javapath','nonjavapath'])
    #         print(len(df2))
    #         df2.drop(list(set(i0)), inplace=True)
    #         print(len(df2))
    #         df2.to_csv('repeattoken.csv', index=False, header=False)
    #     if filename == 'repeattoken.csv':
    #         print('repeatmatchless.csv')
    #         df2 = read_csv('repeatmatchless.csv', ['field','text1', 'text2', 'label', 'javaclass', 'nonjavafile','nonjavafilename','javapath','nonjavapath'])
    #         print(len(df2))
    #         df2.drop(list(set(i0)), inplace=True)
    #         print(len(df2))
    #         df2.to_csv('repeatmatchless.csv', index=False, header=False)
for i in ['itracker','powerstone']:#
    os.chdir("E:\\nuaa\\1st_Year\\1_code\\LocalGit\\bert\\"+i)
    for filename in ['uniquematch2.csv']:
        df = read_csv(filename, ['text2', 'label'])
        train1 = df.loc[df['label'] == str(1), ['text2', 'label']]
        train0 = df.loc[df['label'] == str(0), ['text2', 'label']]
        i0 = []
        for index, row in train1.iterrows():
            text2 = row['text2']
            for index0, row0 in train0.loc[train0['text2'] == text2].iterrows():
                if text2 == row0['text2']:
                    i0.append(index0)
        df.drop(list(set(i0)), inplace=True)
        df.to_csv(filename, index=False, header=False)
