import re
from bs4 import BeautifulSoup
def get_token(tld,classname):
    token = []
    soup = BeautifulSoup(open(tld, encoding='utf-8'), 'lxml')
    #print(soup.prettify())
    e=soup.find("tag-class",string=re.compile(classname))
    #print(e.string.replace('\n','').lstrip())
    if e is not None:
        for i in e.string.replace('\n', '').strip().split('.'):
            token.append(i)
        # print(e.parent)
        for child in e.parent.children:
            if child.name == 'name':
                token.append(child.string)
            if child.name == 'attribute':
                for c in child.children:
                    if c.name == 'name':
                        token.append(c.string)
    else:
        e = soup.find("tagclass", string=re.compile(classname))
        for i in e.string.replace('\n', '').strip().split('.'):
            token.append(i)
        # print(e.parent)
        for child in e.parent.children:
            if child.name == 'name':
                token.append(child.string)
            if child.name == 'attribute':
                for c in child.children:
                    if c.name == 'name':
                        token.append(c.string)
    print(token)
    return token
if __name__ == '__main__':
    tld=r"E:\nuaa\1st_Year\1_code\LocalGit\itracker\itracker-web\src\main\webapp\WEB-INF\tld\itracker.tld"
    get_token(tld,"FormatLinkTag")