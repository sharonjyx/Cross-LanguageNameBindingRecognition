import pandas as pd
from gettoken import parseTLD, parseJSP, parseXML, parseHTML


#print(df.head())
#print(type(np.array(df)))
class Match:
    fieldname=''
    fieldclass=''
    fieldpath=''
    line=''
    number=''
    filename=''
    filepath=''
    def __init__(self,fieldname,fieldclass,fieldpath,line,number,filename,filepath):
        self.fieldname=fieldname
        self.fieldclass=fieldclass
        self.fieldpath=fieldpath
        self.line=line
        self.number=number
        self.filename=filename
        self.filepath=filepath

list=[]
newlist=[]
def read(path):
    pd.set_option('display.max_columns', 10)
    df = pd.read_csv(path, header=None)
    for tup in df.itertuples():
        fieldpath="E:\\nuaa\\1st_Year\\1_code\\LocalGit\\"+tup[3].replace("\\\\","\\")
        filepath="E:\\nuaa\\1st_Year\\1_code\\LocalGit\\"+tup[7].replace("\\\\","\\")
        line=tup[4].replace("\"\"","\"")
        #print(tup)
        m = Match(tup[1], tup[2], fieldpath, line, tup[5], tup[6], filepath)
        list.append(m)
    # print(list)
    for i in list:
        print(i.fieldname)
        print(i.line)
        token=[]
        if (i.filename.endswith(".html")):
            token=html(i)
        elif(i.filename.endswith(".xml")):
            token=xml(i)
        elif (i.filename.endswith(".jsp")):
            #print("jsp")
            token=jsp(i)
        elif (i.filename.endswith(".tld")):
            token=tld(i)
        if token==[]:
            print('error:'+i.fieldname+' '+i.line+str(i.number))
        # if len(token)<7:
        #     l = re.sub(r'[\W_]+', ' ', i.line)
        #     for i in l.split(' '):
        #         if i!='':
        #             token.append(i)
        #     #print(l)
        #     print(token)
        newlist.append(token)
    return newlist
def html(i):
    token=[]
    fieldname = i.fieldname
    filepath = i.filepath
    line=i.line
    if 'data-th-' in line:
        token= parseHTML.get_token1(filepath, line, fieldname)
    elif 'wicket:id'in line:
        token= parseHTML.get_token2(filepath)
    elif 'ognl' in line or 'jwcid' in line:
        token= parseHTML.get_token3(filepath, line)
    return token
def xml(i):
    token=[]
    fieldname = i.fieldname
    fieldclass=i.fieldclass
    line = i.line
    filepath = i.filepath
    filename=i.filename
    if 'mall'in filepath:
        if 'record.'+fieldname in line:
            token = parseXML.get_token7_1(filepath, fieldname)
        elif 'item.'+fieldname in line:
            token = parseXML.get_token7_2(filepath)
        elif '<property name=' in line:
            token = parseXML.get_token8(filepath, fieldname)
        else:
            token = parseXML.get_token6(filepath, fieldname)
    else:
        for i in ['class', 'id', 'version', 'many-to-one', 'property', 'set', 'bag', 'one-to-one']:
            if '<' + i in line and '.hbm.xml' in filename:
                token = parseXML.get_token2(filepath)
                if token != []:
                    return token
            elif 'name=' in line and '.hbm.xml' in filename:
                token = parseXML.get_token2(filepath)
                if token != []:
                    return token
        if '<field property' in line:
            token = parseXML.get_token3(filepath, fieldname)
        elif '<property name' in line:
            token = parseXML.get_token14(filepath, fieldname, fieldclass)
        elif '<param-name>' in line:
            token = parseXML.get_token5(filepath, fieldname)
        elif 'p:' + fieldname in line:
            # print('p:')
            token = parseXML.get_token1(filepath, line, fieldname)
        elif 'menu id' in line:
            token = parseXML.get_token12(filepath, fieldname)
        elif 'beans:property name' in line:
            token = parseXML.get_token13(filepath, fieldname)
        elif '<issue-' + fieldname + '>' in line or '</issue-' + fieldname + '>' in line or '<owner>' in line or '<creator>' in line:
            token = parseXML.get_token4_1(filepath, fieldname)
        elif '<project-' + fieldname + '>' in line or '</project-' + fieldname + '>' in line or '<versions>' in line or '</versions>' in line or '</components>' in line or '<components>' in line:
            token = parseXML.get_token4_2(filepath, fieldname)
        elif '<cache name' in line:
            token = parseXML.get_token10(filepath, line, fieldname)
        elif '.' + fieldname in line or 'where' in line or fieldname + '.' in line:
            # print("token9")
            token = parseXML.get_token9(filepath, line, fieldname, filename)
        elif fieldname + '=' in line:
            token = parseXML.get_token11(filepath, fieldname)
    return token
def jsp(i):
    token = []
    fieldname = i.fieldname
    line = i.line
    filepath = i.filepath
    #print(fieldname)
    if 'jrecruiter'in path:
        if 'if test=' in line or'items='in line:
            token = parseJSP.get_token8(filepath, line)
        elif 'list=' in line:
            token = parseJSP.get_token7(filepath, fieldname)
        elif fieldname+'.'in line or '.'+fieldname in line:
            token = parseJSP.get_token10(filepath, line, fieldname)
        elif 'property='in line:
            token= parseJSP.get_token11(filepath)
    elif 'property='in line:
        token= parseJSP.get_token1(filepath)
    elif '.'+fieldname in line:
        #token=parseJSP.get_token2(filepath,line,fieldname)
        #print('get_token5')
        token = parseJSP.get_token5(filepath, line, fieldname)
    elif 'value='in line or 'href=${'in line or '${' in line:
        token= parseJSP.get_token9(filepath, fieldname)
    elif fieldname+'=' in line:
        #print("token3")
        token= parseJSP.get_token3(filepath, line, fieldname)
    elif '<bean:parameter'in line:
        token= parseJSP.get_token8(filepath, fieldname)
    elif 'path='in line:
        token= parseJSP.get_token4(filepath, fieldname)
    return token
def tld(i):
    fieldclass = i.fieldclass.replace('.java','')
    filepath = i.filepath
    token= parseTLD.get_token(filepath, fieldclass)
    return token
def write(list):
    file_handle = open(
        r'E:\nuaa\1st_Year\1_code\test\match\dataset\mall\mall_Repeat_token.csv',
        mode='w', encoding='utf-8')
    print(len(list))
    for token in list:
        newlist = []
        line = ''
        for ss in token:
            for s in ss.split(' '):
                if s not in newlist and s!='':
                    newlist.append(s)
                    line=line+' '+s
        file_handle.write(line.strip() + '\n')
    print("-------------------------")
if __name__ == '__main__':
    path=r'E:\nuaa\1st_Year\1_code\test\match\dataset\mall\mall_Repeat_match.csv'
    #read(path)
    write(read(path))