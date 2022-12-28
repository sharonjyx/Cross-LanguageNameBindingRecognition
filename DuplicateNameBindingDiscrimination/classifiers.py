import os
from sklearn.preprocessing import MinMaxScaler
import warnings

from feature.smote import stable_SMOTE

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random
import time
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import datetime
from sklearn.model_selection import train_test_split
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
def read_csv(input_file,columns):
    with open(input_file,"r",encoding="utf-8") as file:
        lines=[]
        for line in file:
            if len(line.strip().split(",")) != 1:
                lines.append(line.strip().split(","))
        df = pd.DataFrame(lines)
        df.columns = columns
    return df
def separate_data(original_data):
    #out-of-sample bootstrap方法产生训练集和测试集
    original_data = np.array(original_data).tolist()
    size = len(original_data)
    train_dataset = []
    train_index = []
    for i in range(size):
        index = random.randint(0, size - 1)
        train_instance = original_data[index]
        train_dataset.append(train_instance)
        train_index.append(index)
    original_index = [z for z in range(size)]
    train_index = list(set(train_index))
    test_index = list(set(original_index).difference(set(train_index)))
    original_data = np.array(original_data)
    train_dataset = original_data[train_index]
    test_dataset = original_data[test_index]
    return train_dataset, test_dataset
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def classifiers(dataset):
    #file = open('classifiers1.csv', mode='a+', newline='')
    #file.write('classifiers,epoch,acc,recall,mcc,precision,F1,auc,total_time'+'\n')
    # 归一化
    dataset = np.array(dataset)
    #print(dataset)
    # dataset=pd.DataFrame(dataset,dtype=np.float)
    scaler = MinMaxScaler(feature_range=[0, 1], copy=False)
    dataset = pd.DataFrame(scaler.fit_transform(dataset))

    tuned_parameterssvm = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
    tuned_parametersknn = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
    tuned_parametersrf = {'n_estimators': [10, 50, 100, 200, 500, 1000]}
    tuned_parametersdt = {'criterion': ('gini', 'entropy'), 'max_depth': [30, 50, 60, 100],
                          'min_samples_leaf': [2, 3, 5, 10]}

    train_dataset,test_dataset=train_test_split(dataset,test_size=0.1,random_state=0)
    train_x = train_dataset.iloc[:, 0:-1].values
    train_y = train_dataset.iloc[:, -1].values


    validation_clf1 = GridSearchCV(svm.SVC(), tuned_parameterssvm, cv=5, n_jobs=-1)
    validation_clf1.fit(train_x, train_y)
    best_parameter = validation_clf1.best_params_
    print(best_parameter)
    print("----------------------------------------------")
    best_c = best_parameter["C"]
    best_kernal = best_parameter["kernel"]

    validation_clf2 = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parametersknn, cv=5, n_jobs=-1)
    validation_clf2.fit(train_x, train_y)
    best_parameter = validation_clf2.best_params_
    print(best_parameter)
    print("----------------------------------------------")
    best_n_neighbors = best_parameter["n_neighbors"]

    validation_clf3 = GridSearchCV(RandomForestClassifier(), tuned_parametersrf, cv=5, n_jobs=-1)
    validation_clf3.fit(train_x, train_y)
    best_parameter = validation_clf3.best_params_
    print(best_parameter)
    print("----------------------------------------------")
    best_n_estimators = best_parameter["n_estimators"]

    validation_clf4 = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parametersdt, cv=5, n_jobs=-1)
    validation_clf4.fit(train_x, train_y)
    best_parameter = validation_clf4.best_params_
    print(best_parameter)
    print("----------------------------------------------")
    best_criterion = best_parameter["criterion"]
    best_max_depth = best_parameter["max_depth"]
    best_min_samples_leaf = best_parameter["min_samples_leaf"]

    clf1 = svm.SVC(C=best_c, kernel=best_kernal,probability=True)
    clf2 = neighbors.KNeighborsClassifier(n_neighbors=best_n_neighbors)
    clf3 = RandomForestClassifier(n_estimators=best_n_estimators)
    clf4 = tree.DecisionTreeClassifier(criterion=best_criterion
                                       , max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf)
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4],
                              use_probas=True,
                              average_probas=False,
                              meta_classifier=lr)
    sclf2 = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4],
                               meta_classifier=lr)
    kf = KFold(n_splits=10, shuffle=True, random_state=5)  # random_state=0
    i = 1
    dataset = pd.DataFrame(dataset)
    t_acc = []
    t_recall = []
    t_mcc = []
    t_precision = []
    t_F1 = []
    t_auc = []
    # 十折交叉验证
    for train_index, test_index in kf.split(dataset):
        total_t = time.time()
        dataset = np.array(dataset)
        # print(test_index.tolist())
        train_dataset = dataset[train_index.tolist()]
        test_dataset = dataset[test_index.tolist()]
        train_dataset = train_dataset[1:, :]
        # print(test_dataset)
        # train_x = train_dataset[1:, 1:-1]
        # train_y = train_dataset[1:, 0]
        stable_smote = stable_SMOTE(5)
        stable_smote_train = stable_smote.fit_sample(train_dataset)
        # print(stable_smote_train)
        if stable_smote_train is False:
            train_dataset = np.array(train_dataset)
            train_x = train_dataset[:, 0:-1]
            train_y = train_dataset[:, -1]
        else:
            stable_smote_train = np.array(stable_smote_train)
            train_x = stable_smote_train[:, 0:-1]
            train_y = stable_smote_train[:, -1]
        # print(train_y)
        test_x = test_dataset[:, 0:-1]
        test_y = test_dataset[:, -1]
        total_acc = []
        total_recall = []
        total_mcc = []
        total_precision = []
        total_F1 = []
        total_auc = []
        for clf, l in zip([clf1, clf2, clf3, clf4, sclf, sclf2],
                          ['SVM', 'KNN', 'Random Forest', 'Decision Tree', 'StackingClassifier',
                           'StackingClassifier2']):
            acc1 = 0
            recall1 = 0
            mcc1 = 0
            precision1 = 0
            F11 =0
            auc1 =0
            for i in range(4):
                print(l)
                clf.fit(train_x, train_y)
                predict_result = clf.predict(test_x)
                # print("predict result: ", predict_result[0])
                # print(time.asctime(time.localtime(time.time())))
                print("---------------------------------")
                true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_y, predict_result,
                                                                                                labels=[0, 1]).ravel()
                recall = true_positive / (true_positive + false_negative)
                mcc = matthews_corrcoef(test_y, predict_result)
                try:
                    auc = roc_auc_score(test_y, predict_result)
                except ValueError:
                    auc = 0
                acc = metrics.accuracy_score(test_y, predict_result)
                precision = true_positive / (true_positive + false_positive)
                import math
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
                acc1+=acc
                recall1 +=recall
                mcc1 +=mcc
                precision1 +=precision
                F11 +=F1
                auc1 +=auc
            acc1 = acc1/4.0
            recall1 = recall1/4.0
            mcc1 = mcc1/4.0
            precision1 = precision1/4.0
            F11 = F11/4.0
            auc1 = auc1/4.0
            print("  Accuracy: {}".format(acc1))
            print("  recall: {}".format(recall1))
            print("  mcc: {}".format(mcc1))
            print("  precision: {}".format(precision1))
            print("  F1: {}".format(F11))
            print("  auc: {}".format(auc1))
            total_acc.append(acc1)
            total_recall.append(recall1)
            total_mcc.append(mcc1)
            total_precision.append(precision1)
            total_F1.append(F11)
            total_auc.append(auc1)
        t_acc.append(total_acc)
        t_recall.append(total_recall)
        t_mcc.append(total_mcc)
        t_precision.append(total_precision)
        t_F1.append(total_F1)
        t_auc.append(total_auc)
    dic = {}
    for i in range(10):
        for j, l in zip([0, 1, 2, 3, 4, 5],
                        ['SVM',
                         'KNN',
                         'Random Forest',
                         'Decision Tree',
                         'StackingClassifier',
                         'StackingClassifier2']):
            if l in dic.keys():
                list1 = dic.get(l)
                acc1 = list1[0] + t_acc[i][j]
                recall1 = list1[1] + t_recall[i][j]
                mcc1 = list1[2] + t_mcc[i][j]
                precision1 = list1[3] + t_precision[i][j]
                F11 = list1[4] + t_F1[i][j]
                auc1 = list1[5] + t_auc[i][j]
                dic[l] = [acc1, recall1, mcc1, precision1, F11, auc1]
            else:
                acc1 = t_acc[i][j]
                recall1 = t_recall[i][j]
                mcc1 = t_mcc[i][j]
                precision1 = t_precision[i][j]
                F11 = t_F1[i][j]
                auc1 = t_auc[i][j]
                dic[l] = [acc1, recall1, mcc1, precision1, F11, auc1]
    for i in ['SVM', 'KNN', 'Random Forest', 'Decision Tree', 'StackingClassifier', 'StackingClassifier2']:
        list2 = dic.get(i)
        total_acc = list2[0] / 10.0
        total_recall = list2[1] / 10.0
        total_mcc = list2[2] / 10.0
        total_precision = list2[3] / 10.0
        total_F1 = list2[4] / 10.0
        total_auc = list2[5] / 10.0
        file = open('classifiers1.csv', mode='a+', newline='')
        file.write(i + ',avg,' + "{}".format(total_acc) + ',' + "{}".format(
            total_recall) + ',' + "{}".format(total_mcc) + ',' + "{}".format(total_precision) + ',' + "{}".format(
            total_F1) + ',' + "{}".format(total_auc) + ',' + "{}".format(format_time(time.time() - total_t)) + '\n')


if __name__ == '__main__':
    for i in ['jrecruiter']:#'itracker', 'sagan', 'springside', 'Tudu-Lists', 'zksample2', 'jrecruiter', 'hispacta', 'powerstone','jtrac', 'mall'
        print(i)
        os.chdir(r"E:\nuaa\1st_Year\1_code\LocalGit\bert\\" + i)
        dataset1 = read_csv('cosine_sim.csv',
                            ['label', 'cosine_sim', 'Euclidean_dist', 'pearson', 'manhattanDisSim', 'cosine_sim1',
                             'Euclidean_dist1', 'lcsSim', 'diceSim', 'editSim', 'levenSim', 'jaroSim',
                             'jaroWinklerSim'])
        dataset2 = read_csv('relationship.csv',['re1', 're2', 're3', 're4', 're5', 're6', 're7', 're8', 're9', 're10', 're11'])
        dataset3 = read_csv('fieldsrelationship.csv', ['javaclassnum', 'fieldsprob', 'fieldsavg', 'fieldssum'])
        dataset = pd.concat([dataset1, dataset2, dataset3], axis=1, ignore_index=True)
        # print(dataset.head())
        dataset.drop([0], inplace=True)
        dataset = dataset.rename(
            columns={0: 'label', 1: 'cosine_sim', 2: 'Euclidean_dist', 3: 'pearson', 4: 'manhattanDisSim',
                     5: 'cosine_sim1',
                     6: 'Euclidean_dist1', 7: 'lcsSim', 8: 'diceSim', 9: 'editSim', 10: 'levenSim', 11: 'jaroSim',
                     12: 'jaroWinklerSim',
                     13: 're1', 14: 're2', 15: 're3', 16: 're4', 17: 're5', 18: 're6', 19: 're7', 20: 're8', 21: 're9',
                     22: 're10', 23: 're11',
                     24: 'javaclassnum', 25: 'fieldsprob', 26: 'fieldsavg', 27: 'fieldssum'}
        )
        column = ['cosine_sim', 'Euclidean_dist', 'pearson', 'manhattanDisSim', 'cosine_sim1', 'Euclidean_dist1',
                  'lcsSim', 'diceSim', 'editSim', 'levenSim', 'jaroSim', 'jaroWinklerSim', 're1', 're2', 're3', 're4',
                  're5', 're6', 're7', 're8', 're9', 're10', 're11', 'javaclassnum', 'fieldsprob', 'fieldsavg',
                  'fieldssum', 'label']
        dataset = dataset[column]
        classifiers(dataset)