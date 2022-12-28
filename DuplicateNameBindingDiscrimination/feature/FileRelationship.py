# -*- coding: utf-8 -*-
# coding: utf-8
import re
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from bs4 import BeautifulSoup
from gettoken import parseXML


def read_csv(input_file,columns):
    with open(input_file,"r",encoding="utf-8") as file:
        lines=[]
        for line in file:
            if len(line.strip().split(",")) != 1:
                lines.append(line.strip().split(","))
        df = pd.DataFrame(lines)
        df.columns = columns
    return df
#xx.hbm.xml——xx.java
def re1(javaclass, nonjavafile):
    soup = BeautifulSoup(open(nonjavafile, encoding='utf-8'), 'xml')
    for e in soup.find_all('class',attrs={'name': True}):
        str=e['name'].split('.')
        if javaclass.lower()==str[len(str)-1].lower():
            return True
    return False
#xx.html——xx.java jtrac
def re2(javaclass,javapath,field):
    f1 = open(javapath, 'rb')
    for i in f1.readlines():
        i = str(i)
        i = i.replace('b\'', '').replace('\\r\\n', '').replace('\'', '').replace('\\\\\\\\', '\\')
        if field in i and javaclass in i:
            return True
    return False
#jsp <html:form action="/" itracker
def re3(javaclass, nonjavafile):
    soup = BeautifulSoup(open(nonjavafile, encoding='utf-8'), 'lxml')
    e=soup.find('html:form',attrs={'action': True})
    #print(e)
    if e is None:
        return False
    str1=e['action']
    #print(str1)
    dicform = parseXML.getdicform()
    xml1 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-config.xml"
    xml2 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-admin-config.xml"
    xml3 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-help-config.xml"
    xml4 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-preferences-config.xml"
    xml5 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-projects-config.xml"
    xml6 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-reports-config.xml"
    xml7 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-searchissues-config.xml"
    for i in [xml1,xml2,xml3,xml4,xml5,xml6,xml7]:
        soup = BeautifulSoup(open(i, encoding='utf-8'), 'lxml')
        e1=soup.find('action', attrs={'path': str1})
        if e1 is not None:
            #print(e1)
            #print(javaclass)
            #print(e1['name'])
            a = dicform.get(e1['name']).split('.')
            #print(a)
            if javaclass.lower() == a[len(a) - 1].lower():
                #print(True)
                return True
    return False
#jsp <form:form commandName="user" Tudu-Lists
def re4(javaclass, nonjavafile,nonjavapath,project):
    soup = BeautifulSoup(open(nonjavapath, encoding='utf-8'), 'lxml')#unicode_escape
    e = soup.find('form:form', attrs={'commandname': True})
    # print(e)
    if e is None:
        return False
    str1 = e['commandname']
    print(str1)
    path1="E:\\nuaa\\1st_Year\\1_code\\test\\"+project+"\\path\\"+project+"javafilepath.txt"
    path=''
    f = open(path1, 'rb')
    newname=''
    for i in nonjavafile.replace('.jsp' ,'').replace('_' ,' ').split(' '):
        newname+=i.title()
    #print(newname)
    for i in f.readlines():
        i = str(i)
        i = i.replace('b\'', '').replace('\\r\\n', '').replace('\'', '').replace('\\\\\\\\', '\\')
        if i.endswith(newname+'Controller.java'):
            path=i
            #print(path)
    if path=='':
        return False
    f1 = open(path, 'rb')
    lines=[]
    for i in f1.readlines():
        i = str(i)
        i = i.replace('b\'', '').replace('\\r\\n', '').replace('\'', '').replace('\\\\\\\\', '\\')
        lines.append(i)
    #print(nonjavafile.replace('.jsp' ,''))
    for i in lines:
        if 'mv.setViewName' in i and nonjavafile.replace('.jsp' ,'') in i:
            #print('mv.setViewName')
            newlist = lines[:lines.index(i)]
            for j in reversed(newlist):
                if  '@RequestMapping' in j:
                    newlist=newlist[newlist.index(j):]
                    break
            for k in newlist:
                if javaclass.lower() in k.lower():
                    return True
        elif 'new ModelAndView'in i and nonjavafile.replace('.jsp' ,'') in i:
            #print('new ModelAndView')
            newlist = lines[:lines.index(i) + 2]
            for j in reversed(newlist):
                if '@RequestMapping' in j:
                    newlist = newlist[newlist.index(j):]
                    break
            for k in newlist:
                if javaclass.lower() in k.lower():
                    return True
        elif '@ModelAttribute' in i and str1 in i:
            #print('@ModelAttribute')
            newlist = lines[lines.index(i):]
            if javaclass.lower() in i.lower():
                return True
    return False
#html sagan
def re5(javaclass, nonjavafilename,nonjavapath,project):
    s=nonjavapath.split('\\')
    s=s[len(s)-2]
    print(s)
    path = "E:\\nuaa\\1st_Year\\1_code\\test\\" + project + "\\path\\" + project + "javafilepath.txt"
    list0 = []
    newlist=[]
    f = open(path, 'rb')
    for i in f.readlines():
        i=str(i)
        i=i.replace('b\'','').replace('\\r\\n','').replace('\'','').replace('\\\\\\\\','\\')
        #print(i)
        if i.endswith('Controller.java'):
            list0.append(i)
    if len(list0)==0:
        return False
    for path in list0:
        f1 = open(path, 'rb')
        line=[]
        for j in f1.readlines():
            j = str(j)
            j = j.replace('b\'', '').replace('\\r\\n', '').replace('\'', '').replace('\\\\\\\\', '\\')
            line.append(j)
        for j in line:
            if 'return'in j and '/'+s+'/'+nonjavafilename.replace('.html' ,'') in j:
                print('/'+s+'/'+nonjavafilename.replace('.html' ,''))
                newlist=line[:line.index(j)]
                for k in list(reversed(newlist)):
                    if 'public' in k:
                        newlist=newlist[newlist.index(k):]
                        break
                for i in newlist:
                    if javaclass.lower() in i.lower():
                        return True
            elif 'return'in j and '/'+nonjavafilename.replace('.html' ,'') in j:
                newlist = line[:line.index(j)]
                for k in reversed(newlist):
                    if 'public' in k:
                        newlist = newlist[newlist.index(k):]
                        break
                for i in newlist:
                    if javaclass.lower() in i.lower():
                        return True
    return False
#nonjavafilename in javapath
def re6(javapath, nonjavafilename):
    f1 = open(javapath, 'rb')
    for i in f1.readlines():
        i = str(i)
        i = i.replace('b\'', '').replace('\\r\\n', '').replace('\'', '').replace('\\\\\\\\', '\\')
        if nonjavafilename in i:
            return True
    return False
#xx.java——xxMapper.xml
def re7(nonjavapath, javaclass):
    soup = BeautifulSoup(open(nonjavapath, encoding='utf-8'), 'xml')
    e=soup.find('resultMap', attrs={'type': True})
    if e is not None and javaclass in e['type']:
        return True
    return False
#record.field item.field
def re8(project,nonjavapath, javaclass,nonjavaline):
    a=''
    if 'record 'in nonjavaline:
        a='@Param("record")'
    elif 'item ' in nonjavaline:
        a='@Param("list")'
    soup = BeautifulSoup(open(nonjavapath, encoding='utf-8'), 'xml')
    e = soup.find('mapper')
    s=e['namespace'].split('.')
    s=s[len(s)-1]+'.java'
    path1 = "E:\\nuaa\\1st_Year\\1_code\\test\\" + project + "\\path\\" + project + "javafilepath.txt"
    path = ''
    f = open(path1, 'rb')
    for i in f.readlines():
        i = str(i)
        i = i.replace('b\'', '').replace('\\r\\n', '').replace('\'', '').replace('\\\\\\\\', '\\')
        if i.endswith(s):
            path = i
            break
    if path == '':
        return False
    f1 = open(path, 'rb')
    for i in f1.readlines():
        i = str(i)
        i = i.replace('b\'', '').replace('\\r\\n', '').replace('\'', '').replace('\\\\\\\\', '\\')
        if a in i and javaclass in i:
            return True
    return False

#spring规则 xml
def re9(javaclass, nonjavapath,field):
    soup = BeautifulSoup(open(nonjavapath, encoding='utf-8'), 'xml')
    for e in soup.find_all('beans:property', attrs={'name': re.compile(field)}):
        if e is not None and javaclass in e.parent['class']:
            return True
    return False
def re10(javaclass, nonjavapath,field):
    soup = BeautifulSoup(open(nonjavapath, encoding='utf-8'), 'xml')
    for e in soup.find_all('property', attrs={'name': re.compile(field)}):
        if e is not None and javaclass in e.parent['class']:
            return True
    return False
def re11(javaclass, nonjavapath, field):
    soup = BeautifulSoup(open(nonjavapath, encoding='utf-8'), 'xml')
    for e in soup.find_all('bean', attrs={'p:'+field+'-ref':True}):
        if e is not None and javaclass in e['class']:
            return True
    return False

class Match:
    field=''
    javaline=''
    nonjavaline=''
    label=''
    javaclass=''
    nonjavafile=''
    nonjavafilename=''
    javapath=''
    nonjavapath=''
    def __init__(self,field,javaline,nonjavaline,label,javaclass,nonjavafile,nonjavafilename,javapath,nonjavapath):
        self.field =field
        self.javaline =javaline
        self.nonjavaline =nonjavaline
        self.label =label
        self.javaclass =javaclass
        self.nonjavafile =nonjavafile
        self.nonjavafilename =nonjavafilename
        self.javapath = javapath
        self.nonjavapath =nonjavapath

if __name__ == '__main__':
    #re3('IssueAttachment', r'E:\nuaa\1st_Year\1_code\LocalGit\itracker\itracker-web\src\main\webapp\portalhome.jsp')
    #'itracker', 'sagan', 'springside', 'Tudu-Lists', 'zksample2'
    for project in ['powerstone']:#'itracker', 'sagan', 'springside', 'Tudu-Lists', 'zksample2','mall','jrecruiter','hispacta','powerstone','jtrac'
        os.chdir(r"E:\nuaa\1st_Year\1_code\LocalGit\bert\\" + project)
        df = read_csv("repeatmatchless.csv", ['field','text1', 'text2', 'label', 'javaclass', 'nonjavafile','nonjavafilename','javapath','nonjavapath'])
        list0 = []
        allre=[]
        for tup in df.itertuples():
            javapath = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\" + tup[8].replace("\\\\", "\\")
            nonjavapath = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\" + tup[9].replace("\\\\", "\\")
            #print(javapath)
            #print(tup)
            m = Match(tup[1],tup[2],tup[3],tup[4],tup[5],tup[6],tup[7],javapath,nonjavapath)
            list0.append(m)
        for i in list0:
            relation = []
            #print(i.nonjavafilename)
            if i.nonjavafilename.endswith(".hbm.xml"):
                if re1(i.javaclass,i.nonjavapath)==True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)
            if i.javaclass==i.nonjavafile and i.nonjavafilename.endswith(".html"):
                if re2(i.javaclass,i.javapath,i.field) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)
            if i.nonjavafilename.endswith(".jsp") and 'itracker' in i.javapath:
                #print(i.nonjavafilename)
                if re3(i.javaclass,i.nonjavapath) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)
            if i.nonjavafilename.endswith(".jsp"):
                #print('re4')
                if re4(i.javaclass, i.nonjavafilename,i.nonjavapath,project) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)
            if i.nonjavafilename.endswith(".html"):
                if re5(i.javaclass, i.nonjavafilename,i.nonjavapath,project) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)
            if re6(i.javapath, i.nonjavafilename) == True:
                relation.append(1)
            else:
                relation.append(0)
            if i.nonjavafilename.endswith("Mapper.xml") and i.javaclass in i.nonjavafile:
                if re7(i.nonjavapath, i.javaclass) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)

            if 'record '+i.field in i.nonjavaline or 'item '+i.field in i.nonjavaline:
                if re8(project,i.nonjavapath, i.javaclass,i.nonjavaline) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)

            if i.nonjavafilename.endswith(".xml") and 'hbm' not in i.nonjavafilename:
                if re9(i.javaclass,i.nonjavapath,i.field) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)
            if i.nonjavafilename.endswith(".xml") and 'hbm' not in i.nonjavafilename:
                if re10(i.javaclass,i.nonjavapath,i.field) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)
            if i.nonjavafilename.endswith(".xml") and 'hbm' not in i.nonjavafilename:
                if re11(i.javaclass,i.nonjavapath,i.field) == True:
                    relation.append(1)
                else:
                    relation.append(0)
            else:
                relation.append(0)

            print(relation)
            allre.append(relation)
        name=['re1','re2','re3','re4','re5','re6','re7','re8','re9','re10','re11']
        df2=pd.DataFrame(columns=name,data=allre)
        df2.to_csv('relationship.csv',index=False)
        print("-------------------------")