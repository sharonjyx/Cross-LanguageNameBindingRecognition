import re
import bs4
from bs4 import BeautifulSoup


def get_text(html):
    word = []
    soup = BeautifulSoup(open(html, encoding='utf-8'), 'lxml')
    # 提取所有的内容
    for str in soup.stripped_strings:
        print(repr(str).replace("'", ""))
        word.append(repr(str).replace("'", ""))
    # print(soup.html.contents)
    for value1 in soup.html.attrs.values():

        if (isinstance(value1, list)):
            for v in value1: word.append(v)
            print(v)
        else:
            word.append(value1)
            print(value1)
    for next_element in soup.body.next_elements:
        if isinstance(next_element, bs4.element.Tag) and next_element != '\n':
            for value in next_element.attrs.values():
                if (isinstance(value, list)):
                    for v in value: word.append(v)
                    print(v)
                else:
                    word.append(value)
                    print(value)
                # print(type(next_element.attrs))
    # print(soup.span.contents)
    return word


def get_new(html):
    newword = []
    for s in get_text(html):
        s = re.sub(r'[\W_]+', ' ', s)
        s = re.sub('[\u4e00-\u9fa5]', '', s)
        list = s.split(' ')
        for l in list:
            if (l.isalpha() and l != '\n'):
                print(l)
                newword.append(l)
    return newword


# 1.	data-th-field
def get_token1(html, line, field):
    token1 = []
    xx = ''
    if line.count('data-th-') == 1:
        for i in line.split(' '):
            if 'data-th-' in i:
                xx = i.split("=")[0]
                break
    else:
        list1 = line.split(' ')
        for i in list1:
            if 'data-th-' in i and field in i:
                xx = i.split("=")[0]
                break
            elif 'data-th-' in i and field not in i:
                list1 = list1[list1.index(i) + 1:]
                for j in list1:
                    if 'data-th-' in j:
                        break
                    if field in j:
                        xx = i.split("=")[0]
    print(xx)
    soup = BeautifulSoup(open(html, encoding='utf-8'), 'lxml')
    if soup.form is not None:
        for value1 in soup.form.attrs.values():
            if (isinstance(value1, list)):
                for v in value1:
                    for x in re.sub(r'[\W_]+', ' ', v).split(' '):
                        if x != '':
                            token1.append(x)
            else:
                for x in re.sub(r'[\W_]+', ' ', value1).split(' '):
                    if x != '':
                        token1.append(x)
                # print(value1)
    for elm in soup.find_all(attrs={xx: True}):
        # print(elm['data-th-field'])
        for x in re.sub(r'[\W_]+', ' ', elm[xx]).split(' '):
            if x != '':
                token1.append(x)
    print(token1)
    return token1


# 2.	wicket:id
def get_token2(html):
    token2 = []
    soup = BeautifulSoup(open(html, encoding='utf-8'), 'lxml')
    for e in soup.find_all(attrs={'wicket:id': True}):
        # print(e['wicket:id'])
        token2.append(e['wicket:id'])
    print(token2)
    return token2


# 3.	hispacta Tapestry
def get_token3(html, line):
    token3 = []
    soup = BeautifulSoup(open(html, encoding='utf-8'), 'lxml')
    # if 'value=' in line:
    for e in soup.find_all(attrs={'value': True}):
        s = e['value'].replace('ognl:components.', '').replace('ognl:item.', '').replace('ognl:page.', '').replace(
            'ognl:', '')
        s = re.sub(r'[\W_]+', ' ', s)
        for i in s.split(' '):
            if i != '':
                token3.append(i)
    if 'field=' in line:
        for e in soup.find_all(attrs={'field': True}):
            s = e['field'].replace('ognl:components.', '').replace('ognl:item.', '')
            # print('2'+s)
            token3.append(s)
    if 'jwcid' in line:
        for e in soup.find_all(attrs={'jwcid': True}):
            if '@' not in e['jwcid'] and '$' not in e['jwcid']:
                # print('3' + e['jwcid'])
                token3.append(e['jwcid'])
    if 'orderProperties' in line:
        for e in soup.find_all(attrs={'orderproperties': True}):
            s = e['orderproperties']
            # print(s)
            s = re.sub(r'[\W_]+', ' ', s)
            for i in s.split(' '):
                if i != '':
                    token3.append(i)
    if 'tag=' in line:
        for e in soup.find_all(attrs={'tag': True}):
            s = e['tag'].replace('tag="ognl:@hispacta.tapestry.page.CrudPage@', '')
            print(s)
            s = re.sub(r'[\W_]+', ' ', s)
            for i in s.split(' '):
                if i != '':
                    token3.append(i)
    for a in ['condition', 'selected', 'source', 'delegate', 'item']:
        if a in line:
            for e in soup.find_all(attrs={a: True}):
                s = e[a].replace('ognl', '').replace('page', '')
                print(s)
                s = re.sub(r'[\W_]+', ' ', s)
                for i in s.split(' '):
                    if i != '':
                        token3.append(i)
    print(token3)
    return token3


def get_new2(w):
    newword = []
    for s in w:
        s = re.sub(r'[\W_]+', '', s)
        if (s != ""):
            # print(s)
            newword.append(s)
    print(newword)
    return newword


if __name__ == '__main__':
    html = r"E:\nuaa\1st_Year\1_code\LocalGit\hispacta\src\main\java\hispacta\tapestry\jwc\CrudPage.html"
    soup = BeautifulSoup(open(html, encoding='utf-8'), 'lxml')
    print(soup.prettify())
