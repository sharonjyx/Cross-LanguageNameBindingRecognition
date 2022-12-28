import os
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
def read_csv(input_file,columns):
    with open(input_file,"r",encoding="utf-8") as file:
        lines=[]
        for line in file:
            if len(line.strip().split(",")) != 1:
                lines.append(line.strip().split(","))
        df = pd.DataFrame(lines)
        df.columns = columns
    return df
def nonjavaword(path):
    f = open(path, 'rb')
    dic={}
    for i in f.readlines():
        i = str(i)
        i = i.replace('b\'', '').replace('\\r\\n', '').replace('\'', '').replace('\\\\\\\\', '\\').replace('\\t','')
        # print(i)
        i =  re.sub(r'[^a-zA-Z0-9_]+', ' ', i)
        #print(i)
        if i!='':
            for s in i.split(' '):
                if s != '':
                    if s not in dic.keys():
                        dic[s] = 1
                    else:
                        dic[s] = dic.get(s) + 1
    return dic
def getfield(alllist):
    all=[]
    for i in alllist:
        newlist=[]
        newlist2=[]
        fields=[]
        javaclass=(i[0])
        path=i[1]
        for a in i[2].split(' '):
            if a!='':
                fields.append(a)
        # print(fields)
        dic=nonjavaword(path)
        if javaclass in dic.keys():
            newlist2.append(dic.get(javaclass))
        else:
            newlist2.append(0)
        for i in fields:
            if i in dic.keys():
                newlist.append(dic.get(i))
            else:
                newlist.append(0)
        # print(newlist)
        prob, avg,sum=process(newlist)
        newlist2.append(prob)
        newlist2.append(avg)
        newlist2.append(sum)
        all.append(newlist2)
    return all
def process(newlist):
    sum=0
    num=0
    length=len(newlist)
    for i in newlist:
        sum += i
        if i!=0:
            num+=1.0
    prob=num/length
    avg=sum/length
    # print(prob)
    # print(avg)
    return prob,avg,sum
if __name__ == '__main__':
    for project in ['itracker', 'sagan', 'springside', 'Tudu-Lists', 'zksample2','mall','jrecruiter','hispacta','powerstone','jtrac']:
        print(project)
        os.chdir(r"E:\nuaa\1st_Year\1_code\LocalGit\bert\\" + project)
        df = read_csv("repeattoken.csv", ['field','text1', 'text2', 'label', 'javaclass', 'nonjavafile','nonjavafilename','javapath','nonjavapath'])
        df2= pd.read_csv("other.csv",header=None)
        javaclasslist=[]
        nonjavapathlist=[]
        fieldslist=[]
        for tup in df.itertuples():
            javaclass = tup[5]
            nonjavapath = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\" + tup[9].replace("\\\\", "\\")
            javaclasslist.append(javaclass)
            nonjavapathlist.append(nonjavapath)
        fieldslist=df2.values.tolist()
        fieldslist = [i for item in fieldslist for i in item]
        alllist=list(zip(javaclasslist,nonjavapathlist,fieldslist))
        #print(alllist)
        newlist=getfield(alllist)
        name = ['javaclassnum', 'fieldsprob', 'fieldsavg', 'fieldssum']
        df2 = pd.DataFrame(columns=name, data=newlist)
        df2.to_csv('fieldsrelationship.csv', index=False)
        print("-------------------------")
    # alllist=[('IpToCountry', 'E:\\nuaa\\1st_Year\\1_code\\LocalGit\\zksample2\\spring-hibernate-backend\\src\\main\\resources\\de\\forsthaus\\backend\\model\\IpToCountry.hbm.xml', 'lVersionUID id ipcIpFrom ipcIpTo ipcCountryCode2 ipcCountryCode3 ipcCountryName version')]
    # print(getfield(alllist))

    #nonjavaword(r"E:\\nuaa\\1st_Year\\1_code\\LocalGit\\zksample2\\spring-hibernate-backend\\src\\main\\resources\\de\\forsthaus\\backend\\model\\IpToCountry.hbm.xml")