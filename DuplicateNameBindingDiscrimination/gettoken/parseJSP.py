import re

import bs4
from bs4 import BeautifulSoup


# 1.	itracker property=" "
def get_token1(jsp):
    token1 = []
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    # print(soup.prettify())
    e = soup.find('html:form')
    # if e is None:
    #     e=soup.find('div',attrs={'action':True})
    for i in e.attrs:
        # print(e[i])
        s = re.sub(r'[\W_]+', ' ', e[i])
        for j in s.split(' '):
            if j != '':
                # print(j)
                token1.append(j)
    for a in soup.find_all(attrs={'property': True}):
        # print(a['property'])
        token1.append(re.sub(r'[\W_]+', '', a['property']))
    print(token1)
    return token1


# 2.	itracker xx.field,need xx.field
def get_token2(jsp, line, field):
    token2 = []
    xx = re.sub(r'[\W_]+', ' ', line)
    xx = re.sub(' +', ' ', xx)
    # print(xx)
    s = xx.replace(field.replace('_', ' '), field).strip().split(' ')
    # print(s)
    index = s.index(field)
    xx = s[index - 1]
    if xx == 'class':
        sss = s[index + 1:]
        index = sss.index(field)
        xx = sss[index - 1]
    # print(xx)
    use(jsp, xx, token2)
    if token2 == []:
        ss = s[index + 1:]
        # print(s)
        if field in ss:
            index = ss.index(field)
            xx = ss[index - 1]
            # print('第二个'+xx)
            use(jsp, xx, token2)
        else:
            token2 = s
    if len(token2) == 1:
        token2 = token2 + s
    print(token2)
    return token2


def use(jsp, xx, token2):
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    e = soup.find('html:form')
    if (e != None):
        for i in e.attrs:
            # print(e[i])
            s = re.sub(r'[\W_]+', ' ', e[i])
            for j in s.split(' '):
                if j != '':
                    # print(j)
                    token2.append(j)
    for next_element in soup.html.next_elements:
        if isinstance(next_element, bs4.element.Tag) and next_element != '\n':
            for i in next_element.attrs:
                l = next_element[i]
                if xx in l:
                    l = re.sub(r'[\W_]+', ' ', l)
                    # print(l)
                    for j in l.split(' '):
                        if j != '' and j != xx:
                            # print(j)
                            token2.append(j)

    return token2


# 3.	itracker tld
def get_token3(jsp, word, field):
    token3 = []
    attr = {}
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    # print(soup.prettify())
    word = word.replace('${ ', '').replace(' }', '').replace('${', '').replace('}', '').replace('<%= ', '').replace(
        '%>', '').replace('<', ' ').replace('/>', '').replace('>', ' ').replace('(', '').replace(')', '')
    # print(word)
    for i in word.split(' '):
        if '=' in i:
            attr[re.sub(r'[\W_]+', '', i.split('=')[0].lower())] = i.split('=')[1].replace('"', '')
    # print(attr)
    # print(field.lower())
    # print(attr.get(field.lower()))
    for e in soup.find_all(attrs={field.lower(): re.compile(attr.get(field.lower()))}):
        # print(e)
        token3.append(e.name.replace('it:', '').replace('html:', ''))
        for i in e.attrs:
            token3.append(i)
    if token3 == [] and field in ['arg0', 'paramValue']:
        e = soup.find('it:formaticonaction', attrs={field.lower(): True})
        if e != None:
            token3.append(e.name.replace('it:', '').replace('html:', ''))
            for i in e.attrs:
                token3.append(i)
    if token3 == [] and len(attr) > 1:
        for key in attr.keys():
            token3.append(key)
    if token3 == [] and 'targetaction' in attr.keys():
        e = soup.find(attrs={field.lower(): re.compile(attr.get(field.lower()).split('.')[0])})
        # print(e)
        if e != None:
            token3.append(e.name.replace('it:', '').replace('html:', ''))
            for i in e.attrs:
                token3.append(i)
    if token3 == [] and 'displaytype' in attr.keys():
        e = soup.find("it:formatcustomfield", attrs={field.lower(): True})
        if e != None:
            token3.append(e.name.replace('it:', '').replace('html:', ''))
            for i in e.attrs:
                token3.append(i)
    print(token3)
    return token3


# 4.	Tudu-Lists path=" "
def get_token4(jsp, field):
    token4 = []
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    for e in soup.find_all(attrs={'path': field}):
        # print(e)
        for i in e.parents:
            if i.name == 'form:form':
                token4.append(i['commandname'])
                for child in i.descendants:
                    if isinstance(child, bs4.element.Tag) and child != '\n':
                        if ('path' in child.attrs):
                            # print(child['path'])
                            token4.append(child['path'])
    print(token4)
    return token4


# 5.	tudu-Lists xx.field,need xx.field
def get_token5(jsp, line, field):
    token5 = []
    for a in line.split('$'):
        if '.' + field in a:
            xx = re.sub(r'[\W_]+', ' ', a)
            xx = re.sub(' +', ' ', xx)
            break
    # print(xx)
    s = xx.replace(field.replace('_', ' '), field).strip().split(' ')
    # print(s)
    index = s.index(field)
    xx = s[index - 1]
    # print(xx)
    soup = BeautifulSoup(open(jsp, encoding='unicode_escape'), 'lxml')
    # print(soup.prettify())
    e = soup.find('form:form')
    if e != None:
        token5.append(e['commandname'])
    else:
        e = soup.html
        # for child in e.descendants:
        #     print(child.name)
        #     for i in child.attrs:
        #         if xx in child.string or xx in child[i]:
        #             print(e['commandname'])
        #             break;
    xx = xx.split('.')[0] + '.'
    token5.append(xx.replace('.', ''))
    if e is None:
        line = line.split('|')[0]
        line = re.sub(r'[\W_]+', ' ', line)
        # print(line)
        for s in line.split(' '):
            if s != '':
                token5.append(s)
        print(token5)
        return token5
    for next_element in e.next_elements:
        if isinstance(next_element, bs4.element.Tag) and next_element != '\n':
            if next_element.string != None and xx in next_element.string:
                l = next_element.string.strip().split('$')[1]
                # l=re.sub(r'[\W_]+', ' ',l)
                if (l.split(".")[1] == xx.replace('.', '')):
                    l = l.split(".")[2]
                else:
                    l = l.split(".")[1]
                l = re.sub(r'[\W_]+', ' ', l)
                for i in l.split(' '):
                    if l != '' and xx != 'workflowMeta.':
                        token5.append(i)
                print(token5)
            for i in next_element.attrs:
                if i != 'key' and xx in next_element[i]:
                    # print(next_element[i].strip())
                    l = next_element[i].strip()
                    if '$' in l:
                        l = l.split('$')[1]
                    l = l.replace('}', '.')
                    # print(l)
                    list = l.split(' ')
                    for j in list:
                        if xx in j:
                            if j.split(".")[1] == xx.replace('.', ''):
                                l = j.split(".")[2]
                            else:
                                l = j.split(".")[1]
                            l = re.sub(r'[\W_]+', ' ', l)
                            # print(l)
                            for i in l.split(' '):
                                if i != '':
                                    token5.append(i)
    for i in token5:
        if i == '' or i == ' ':
            token5.remove(i)
    print(token5)
    return token5


# 6.	jrecruiter items="${field}"
def get_token6(jsp, field):
    token6 = []
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    for e in soup.find_all('jmesa:tablefacade', attrs={'items': re.compile(field)}):
        for i in e.attrs:
            if e[i] != '':
                l = re.sub(r'[\W_]+', '', e[i])
                token6.append(l)
    print(token6)
    return token6


# 7.	jrecruiter list="field"
def get_token7(jsp, field):
    token7 = []
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    for e in soup.find_all(attrs={'list': re.compile(field)}):
        for i in e.attrs:
            if e[i] != '':
                for a in e[i].split('.'):
                    if a != '' and a.isalpha():
                        l = re.sub(r'[\W_]+', ' ', a)
                        token7.append(l)
    print(token7)
    return token7


# 8.	jrecruiter if test=/items=
def get_token8(jsp, line):
    token8 = []
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    # print(soup.prettify())
    for a in ['id', 'action', 'test']:
        for e in soup.find_all(attrs={a: True}):
            s = e[a]
            # print(s)
            s = re.sub(r'[\W_]+', ' ', s)
            for i in s.split(' '):
                if i != '':
                    token8.append(i)
    if 'items=' in line:
        e1 = soup.find('jmesa:tablefacade')
        for attr in e1.attrs:
            # print(e1[attr])
            for i in re.sub(r'[\W_]+', ' ', e1[attr]).split(' '):
                if i != '':
                    token8.append(i)
    print(token8)
    return token8


# 9.springside value=${field} or ${} or value="field" or href=${field}
def get_token9(jsp, field):
    token9 = []
    soup = BeautifulSoup(open(jsp, encoding='unicode_escape'), 'lxml')
    # print(soup.prettify())
    e = soup.find('form')
    if e != None:
        if 'id' in e.attrs:
            token9.append(e['id'])
        if 'name' in e.attrs:
            token9.append(e['name'])
    else:
        e = soup.html
    for element in e.next_elements:
        if isinstance(element, bs4.element.Tag) and element != '\n':
            for i in element.stripped_strings:
                if '${' in i:
                    for x in re.sub(r'[\W_]+', ' ', i).split(' '):
                        if x != '' and IsnotChinese(x):
                            token9.append(x)
    for ele in soup.find_all(attrs={'value': True}):
        for xx in re.sub(r'[\W_]+', ' ', ele['value']).split(' '):
            if xx != '' and IsnotChinese(xx):
                token9.append(xx)
    for ele2 in soup.find_all('a', attrs={'href': re.compile(field)}):
        for xx2 in re.sub(r'[\W_]+', ' ', ele2['href']).split(' '):
            if xx2 != '' and IsnotChinese(xx2):
                token9.append(xx2)
    print(token9)
    return token9


# 10.jrecruiter xx.field=/field.
def get_token10(jsp, line, field):
    token10 = []
    if '.' + field in line:
        xx = re.sub(r'[\W_]+', ' ', line)
        xx = re.sub(' +', ' ', xx)
        # print(xx)
        s = xx.replace(field.replace('_', ' '), field).strip().split(' ')
        # print(s)
        index = s.index(field)
        xx = s[index - 1] + '.'
    else:
        xx = field + '.'
    print(xx)
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    for next_element in soup.html.next_elements:
        if isinstance(next_element, bs4.element.Tag) and next_element != '\n':
            for i in next_element.attrs:
                l = next_element[i]
                while xx in l:
                    # print(l)
                    index = l.index(xx)
                    l1 = l[index:].split(' ')[0]
                    l1 = re.sub(r'[\W_]+', ' ', l1).split(' ')[1]
                    # print(l)
                    token10.append(l1)
                    l = l[index + len(xx):]
            for j in next_element.stripped_strings:
                while xx in j:
                    # print(l)
                    index = j.index(xx)
                    j1 = j[index:].split(' ')[0]
                    j1 = re.sub(r'[\W_]+', ' ', j1).split(' ')[1]
                    # print(j1)
                    token10.append(j1)
                    j = j[index + len(xx):]
                    # print(j)
    # print(token10)
    if len(set(token10)) < 2:
        for e in soup.find_all(attrs={'action': True}):
            for a in re.sub(r'[\W_]+', ' ', e['action']).split(' '):
                token10.append(a)
        for next_element in soup.html.next_elements:
            if isinstance(next_element, bs4.element.Tag) and next_element != '\n':
                for i in next_element.attrs:
                    l = next_element[i]
                    if '${' in l:
                        index = l.index('$')
                        l = l[index:]
                        for l1 in re.sub(r'[\W_]+', ' ', l).split(' '):
                            if l1 != '':
                                token10.append(l1)
                for j in next_element.stripped_strings:
                    if '${' in j:
                        # print(l)
                        index = j.index('$')
                        j = j[index:]
                        for j1 in re.sub(r'[\W_]+', ' ', j).split(' '):
                            if j1 != '':
                                token10.append(j1)
    print(token10)
    return token10


# 11.jrecruiter <jmesa:htmlColumn property=
def get_token11(jsp):
    token11 = []
    soup = BeautifulSoup(open(jsp, encoding='utf-8'), 'lxml')
    for e in soup.find_all('jmesa:htmlcolumn', attrs={'property': True}):
        s = e['property']
        # print(s)
        s = re.sub(r'[\W_]+', ' ', s)
        for i in s.split(' '):
            if i != '':
                token11.append(i)

    print(token11)
    return token11


def IsnotChinese(character):
    '''判断是否为中文字符'''
    for cha in character:
        if '\u0e00' <= cha <= '\u9fa5':
            return False
    else:
        return True


if __name__ == '__main__':
    jsp = r'E:\nuaa\1st_Year\1_code\LocalGit\jrecruiter\jrecruiter-web\src\main\webapp\WEB-INF\jsp\admin\includes\job_location.jsp'
    soup = BeautifulSoup(open(jsp, encoding='unicode_escape'), 'lxml')
    print(soup.prettify())
    get_token7(jsp, 'regions')
