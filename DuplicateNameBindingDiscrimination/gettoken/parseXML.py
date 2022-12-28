import re
import bs4
from bs4 import BeautifulSoup
def get_strings(xml):
    word = []
    soup = BeautifulSoup(open(xml), 'lxml')
    #print(soup.prettify())
    # 提取所有的内容
    for str in soup.stripped_strings:
        #print(repr(str).replace("'", ""))
        word.append(repr(str).replace("'", ""))
    print(word)
    return word
#存储formbean的字典
def get_form(xml):
    form_bean = {}
    soup = BeautifulSoup(open(xml), 'lxml')
    for e in soup.find_all('form-bean', attrs={'name': True}):
        #print(e['name'])
        #print(e['type'])
        form_bean[e['name']]=e['type']
    #print(form_bean)
    return form_bean
def getdicform():
    dicform={}
    xml1 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-config.xml"
    xml2 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-admin-config.xml"
    xml3 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-help-config.xml"
    xml4 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-preferences-config.xml"
    xml5 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-projects-config.xml"
    xml6 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-reports-config.xml"
    xml7 = "E:\\nuaa\\1st_Year\\1_code\\LocalGit\\itracker\\itracker-web\\src\\main\\webapp\\WEB-INF\\config\\struts-module-searchissues-config.xml"
    dicform.update(get_form(xml1))
    dicform.update(get_form(xml2))
    dicform.update(get_form(xml3))
    dicform.update(get_form(xml4))
    dicform.update(get_form(xml5))
    dicform.update(get_form(xml6))
    dicform.update(get_form(xml7))
    #print(dicform)
    return dicform
#1.	Spring zksample2 p:field
def get_token1(xml,line,field):
    token1 = []
    for i in line.split(' '):
        if 'p:'+field in i:
            xx=i
            break
    #print(xx)
    attr=xx.split('=')[0]
    #print(attr)
    value=xx.split('=')[1].replace('>','').replace('\"','')
    #print(value)
    soup = BeautifulSoup(open(xml,encoding='utf-8'), 'xml')
    e=soup.find('bean', attrs={attr:value})
    if 'id' in e.attrs:
        token1.append(e['id'])
    if 'name' in e.attrs:
        token1.append(e['name'])
    for i in e['class'].split('.'):
        if i != '':
            # print(i)
            token1.append(i)
    for a in e.attrs:
        if a != 'id' and a != 'class' and 'p:' in a:
            a = re.sub(r'[\W_]+', ' ', a).replace('p', '').replace('ref', '').strip()
            token1.append(a)
    for child in e.children:
        if isinstance(child, bs4.element.Tag) and child != '\n':
            # print(child['name'])
            token1.append(child['name'])
    print(token1)
    return token1
#Spring springside <property name="sql">
def get_token11(xml,field):
    token1=[]
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'xml')
    for e in soup.find_all('property', attrs={'name': field}):
        # print(e.parent['id'])
        token1.append(e.parent['id'])
        for l in e.parent['class']:
            l = re.sub(r'[\W_]+', ' ', l)
            for i in l.split(' '):
                # print(i)
                token1.append(i)
        for i in soup.select('bean[id=' + e.parent['id'] + ']'):
            for child in i.children:
                if isinstance(child, bs4.element.Tag) and child != '\n':
                    # print(child['name'])
                    token1.append(child['name'])
    print(token1)
    return token1
#2.	Hibernate
def get_token2(xml):
    token2 = []
    soup = BeautifulSoup(open(xml,encoding='utf-8'), 'lxml')
    for i in ['class','id','version','many-to-one','property','set','bag','one-to-one']:
        for e in soup.find_all(i,attrs={'name': True}):
            #print(e['name'])
            for i in re.sub(r'[\W_]+', ' ', e['name']).split(' '):
                token2.append(i)
    print(token2)
    return token2
#3.	Struts
def get_token3(xml,field):
    token3 = []
    dicform=getdicform()
    soup = BeautifulSoup(open(xml,encoding='utf-8'), 'xml')
    for e in soup.find_all('field', attrs={'property': field}):
        #print(e.parent['name'])
        token3.append(e.parent['name'])
        #print(e.parent['name'])
        #token3.append(dicform.get(e.parent['name']))get()
        l= dicform.get(e.parent['name'])
        if l is not None:
            l = re.sub(r'[\W_]+', ' ', l)
            for i in l.split(' '):
                # print(i)
                token3.append(i)
            for i in soup.select('form[name=' + e.parent['name'] + ']'):
                for child in i.children:
                    if isinstance(child, bs4.element.Tag) and child != '\n':
                        # print(child['property'])
                        token3.append(child['property'])
        else:
            for child in e.parent.children:
                if isinstance(child, bs4.element.Tag) and child != '\n':
                    # print(child['property'])
                    token3.append(child['property'])
    print(token3)
    return token3
#4.1<issue-field>标签<owner><creator>
def get_token4_1(xml,field):
    token4 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'lxml')
    s='issue-'+field
    e=soup.find(s)
    if e is None:
        s=field
        e = soup.find(s)
    #print(e.parent)
    for child in e.parent.children:
        if isinstance(child, bs4.element.Tag) and child != '\n' and 'issue' in child.name:
            #print(child.name.replace('issue-',''))
            token4.append(child.name.replace('issue-',''))
    print(token4)
    return token4
#4.2<project-field>,<versions>,<components>
def get_token4_2(xml,field):
    token4 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'lxml')
    s='project-'+field
    e=soup.find(s)
    if e is None:
        s=field
        e = soup.find(s)
    #print(e.parent)
    for child in e.parent.children:
        if isinstance(child, bs4.element.Tag) and child != '\n':
            #print(child.name.replace('project-',''))
            token4.append(child.name.replace('project-',''))
    print(token4)
    return token4
#5.	springside Fileter
def get_token5(xml,field):
    token5 = []
    soup = BeautifulSoup(open(xml,encoding='utf-8'), 'lxml')
    for e in soup.find_all(string=field):
        #print(e)
        #print(e.parent.parent.parent.name)
        for str in e.parent.parent.parent.stripped_strings:
            str=repr(str).replace("'", "").replace("-", "").replace(".", " ")
            str=re.sub(r'[\W_]+', ' ', str)
            for i in str.split(' '):
                #print(i)
                if i!=''and i.isalpha():
                    token5.append(i)
    print(token5)
    return token5
#6.	mall MyBatis
def get_token6(xml,field):
    token6 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'xml')
    for a in ['result','association','collection','id']:
        for e in soup.find_all(a, attrs={'property': field}):
            if e is not None:
                # print(e)
                # print(e.parent.name)
                if 'id' in e.parent.attrs:
                    token6.append(e.parent['id'])
                if 'type' in e.parent.attrs:
                    for i in e.parent['type'].split('.'):
                        token6.append(i)
                for child in e.parent.children:
                    if isinstance(child, bs4.element.Tag) and child != '\n':
                        token6.append(child['property'])
    if len(token6)==0:
        e = soup.find('mapper', attrs={'namespace': True})
        for i in e['namespace'].split('.'):
            token6.append(i)
        for next_element in e.next_elements:
            if isinstance(next_element, bs4.element.Tag) and next_element != '\n':
                for i in next_element.attrs:
                    for j in re.sub(r'[\W_]+', ' ', next_element[i]).split(' '):
                        token6.append(j)
    print(token6)
    return token6
#7.1	mall MyBatis "record.field"
def get_token7_1(xml,field):
    token7 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'lxml')
    e=soup.find('mapper')
    for i in e['namespace'].split('.'):
        token7.append(i)
    for next_element in e.next_elements:
        if isinstance(next_element, bs4.element.Tag) and next_element != '\n':
            for j in next_element.stripped_strings:
                while 'record.' in j:
                    #print(j)
                    index = j.index('record.')
                    j1 = j[index:].split(' ')[0]
                    j1 = re.sub(r'[\W_]+', ' ', j1).split(' ')[1]
                    #print(j1)
                    token7.append(j1)
                    j = j[index + len('record.'):]
    for e in soup.find_all('if', attrs={'test': re.compile('record.'+field)}):
        #print(e)
        #print(e.parent.parent)
        token7.append(e.parent.parent['id'])
        for child in e.parent.children:
            if isinstance(child, bs4.element.Tag) and child != '\n':
                # print(child['test'].split('!')[0].replace('record.','').replace(' ',''))
                token7.append(child['test'].split('!')[0].replace('record.', '').replace(' ', ''))
    print(token7)
    return token7
#7.2	mall MyBatis test="item.field"
def get_token7_2(xml):
    token7 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'lxml')
    for i in soup.find('mapper')['namespace'].split('.'):
        token7.append(i)
    e=soup.find('mapper')
    for next_element in e.next_elements:
        if isinstance(next_element, bs4.element.Tag) and next_element != '\n':
            for j in next_element.stripped_strings:
                while 'item.' in j:
                    #print(j)
                    index = j.index('item.')
                    j1 = j[index:].split(' ')[0]
                    #print(j1)
                    j1 = re.sub(r'[\W_]+', ' ', j1).split(' ')[1]
                    token7.append(j1)
                    j = j[index + len('item.'):]
    print(token7)
    return token7
#8.	mall <property name=
def get_token8(xml,field):
    token8 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'lxml')
    for e in soup.find_all('property', attrs={'name': field}):
        # print(e)
        # print(e.parent.name)
        for i in e.parent['type'].split('.'):
            token8.append(i)
        for child in e.parent.children:
            if isinstance(child, bs4.element.Tag) and child != '\n':
                token8.append(child['name'])
    print(token8)
    return token8
def get_token9(xml,line,field,filename):
    token9 = []
    filename=filename.replace('.hbm.xml', '')
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'xml')
    tokenx=get_token2(xml)
    #print(line)
    if field in tokenx:
        token9=tokenx
    else:
        line.replace('*',',')
        for e in soup.find_all('query'):
            for s in e.stripped_strings:
                if line==s:
                    break
        for s in e.stripped_strings:
            s = re.sub(r'[\W_]+', ' ', s)
            # print(s)
            for i in s.split(' '):
                if i not in ['SELECT', 'FROM', 'JOIN', 'AS', 'WHERE'] and i != '':
                    token9.append(i)
    # for i in line.split(' '):
    #     #print(i)
    #     if field in i:
    #         print(i.split('.')[0])
    #         if filename==i.split('.')[0]:
    #             token9=get_token2(xml)
    #         else :
    #             token9=get_token2(xml.replace(filename,i.split('.')[0]))
    print(token9)
    return token9
#tudu-Lists springside <cache name=xx.field
def get_token10(xml,line,field):
    token10 = []
    xx = re.sub(r'[\W_]+', ' ', line)
    xx = re.sub(' +', ' ', xx)
    # print(xx)
    s = xx.replace(field.replace('_', ' '), field).strip().split(' ')
    # print(s)
    index = s.index(field)
    xx = s[index - 1]
    print(xx)
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'lxml')
    for e in soup.find_all('cache', attrs={'name': re.compile(xx)}):
        # print(e.parent['id'])
        if xx+'.' in e['name']:
            for i in e['name'].split('.'):
                token10.append(re.sub(r'[\W_]+', ' ', i).strip())
    print(token10)
    return token10
#zksample2 menuItem field=
def get_token11(xml,field):
    token11 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'xml')
    token11.append('menuItem')
    for e in soup.find_all('menuItem', attrs={field:True}):
        for i in e.attrs:
            if i not in token11:
                token11.append(i)
    print(token11)
    return token11
#zksample2 <menu field=
def get_token12(xml,field):
    token12 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'xml')
    token12.append('menu')
    for e in soup.find_all('menu', attrs={field:True}):
        for i in e.attrs:
            if i not in token12:
                token12.append(i)
    print(token12)
    return token12
#zksample2 beans:property name=field
def get_token13(xml,field):
    token13 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'xml')
    for e in soup.find_all('beans:property', attrs={'name':re.compile(field)}):
        token13.append(e.parent['id'].replace('\"',''))
        for i in re.sub(r'[\W_]+', ' ', e.parent['class']).split(' '):
            if i!=' ':
                token13.append(i)
        for child in e.parent.children:
            if isinstance(child, bs4.element.Tag) and child != '\n':
                #print(child['name'])
                token13.append(child['name'])
    print(token13)
    return token13
#zksample2 springside jtrac<bean> <property name=field
def get_token14(xml,field,fieldclass):

    token14 = []
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'xml')
    #print(fieldclass.replace('.java',''))
    e=soup.find('bean', attrs={'class':re.compile(fieldclass.replace('.java',''))})
    if fieldclass=='AbstractMultiActionController.java':
        if e is None:
            fieldclass='DefaultMultiActionController.java'
            e = soup.find('bean', attrs={'class': re.compile(fieldclass.replace('.java', ''))})
        if e is None:
            fieldclass='RestMultiActionController.java'
            e = soup.find('bean', attrs={'class': re.compile(fieldclass.replace('.java', ''))})
    if 'id' in e.attrs:
        token14.append(e['id'].replace('\"', ''))
    if 'name' in e.attrs:
        token14.append(e['name'].replace('\"', ''))
    for i in re.sub(r'[\W_]+', ' ', e['class']).split(' '):
        if i != ' ':
            token14.append(i)
    for child in e.children:
        if isinstance(child, bs4.element.Tag) and child != '\n':
            # print(child['name'])
            token14.append(child['name'])
    print(token14)
    return token14
if __name__ == '__main__':
    xml = r'E:\nuaa\1st_Year\1_code\LocalGit\powerstone\web\construction.jsp'
    soup = BeautifulSoup(open(xml, encoding='utf-8'), 'lxml')
    print(soup.prettify())
    #get_token7_2(xml,'orderId')