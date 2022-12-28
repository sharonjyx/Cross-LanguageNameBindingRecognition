from collections import Counter
import numpy as np
import pandas as pd
class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest
    def fit_sample(self, x_dataset):
        x_dataset = pd.DataFrame(x_dataset)
        #print(x_dataset )
        # x_dataset = x_dataset.rename(
        #     columns={0:'cosine_sim', 1:'Euclidean_dist', 2:'pearson', 3:'manhattanDisSim', 4:'cosine_sim1',
        #      5:'Euclidean_dist1', 6:'lcsSim', 7:'diceSim', 8:'editSim', 9:'levenSim',10: 'jaroSim',11: 'jaroWinklerSim',
        #     12:'re1',13: 're2',14: 're3',15: 're4', 16:'re5', 17:'re6', 18:'re7',19: 're8', 20:'re9', 21:'re10', 22:'re11',
        #     23:'javaclassnum', 24:'fieldsprob',25: 'fieldsavg', 26:'fieldssum',27: 'label'}
        #    )
        x_dataset = x_dataset.rename(
                columns={0:'cosine_sim1', 1:'Euclidean_dist1', 2:'lcsSim', 3:'diceSim', 4:'editSim', 5:'levenSim',6: 'jaroSim',7: 'jaroWinklerSim',
                8:'re1',9: 're2',10: 're3',11: 're4', 12:'re5', 13:'re6', 14:'re7',15: 're8', 16:'re9', 17:'re10', 18:'re11',
                19:'javaclassnum', 20:'fieldsprob',21: 'fieldsavg', 22:'fieldssum',23: 'label'}
               )
        #print(x_dataset)
        total_pair = []
        # print(k_nearest)
        defective_instance = x_dataset[x_dataset['label']>0]
        clean_instance = x_dataset[x_dataset['label'] == 0]
        defective_number = len(defective_instance)
        #print(defective_number)
        clean_number = len(clean_instance)
        target_defect_ratio = 0.5
        need_number = int((target_defect_ratio * len(x_dataset) - defective_number) / (1 - target_defect_ratio))  # clean_number - defective_number
        print(need_number)
        # print(clean_number - defective_number)
        # exit()
        if need_number <= 0:
            return False
        generated_dataset = []
        synthetic_dataset = pd.DataFrame()
        number_on_each_instance = need_number / defective_number  # 每个实例分摊到了生成几个的任务
        total_pair = []

        rround = number_on_each_instance / self.z_nearest
        while rround >= 1:
            for index, row in defective_instance.iterrows():
                temp_defective_instance = defective_instance.copy(deep=True)
                subtraction = row - temp_defective_instance
                square = subtraction ** 2
                row_sum = square.apply(lambda s: s.sum(), axis=1)
                distance = row_sum ** 0.5
                temp_defective_instance["distance"] = distance
                temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
                neighbors = temp_defective_instance[1:self.z_nearest + 1]
                for a, r in neighbors.iterrows():
                    selected_pair = [index, a]
                    selected_pair.sort()
                    total_pair.append(selected_pair)
            rround = rround - 1
        need_number1 = need_number - len(total_pair)
        number_on_each_instance = need_number1 / defective_number

        for index, row in defective_instance.iterrows():
            temp_defective_instance = defective_instance.copy(deep=True)
            subtraction = row - temp_defective_instance
            square = subtraction ** 2
            row_sum = square.apply(lambda s: s.sum(), axis=1)
            distance = row_sum ** 0.5

            temp_defective_instance["distance"] = distance
            temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
            neighbors = temp_defective_instance[1:self.z_nearest + 1]
            neighbors = neighbors.sort_values(by="distance", ascending=False)  # 这里取nearest neighbor里最远的
            target_sample_instance = neighbors[0: int(number_on_each_instance)]
            target_sample_instance = target_sample_instance.drop(columns="distance")
            for a, r in target_sample_instance.iterrows():
                selected_pair = [index, a]
                selected_pair.sort()
                total_pair.append(selected_pair)
        temp_defective_instance = defective_instance.copy(deep=True)
        residue_number = need_number - len(total_pair)
        residue_defective_instance = temp_defective_instance.sample(n=residue_number)
        for index, row in residue_defective_instance.iterrows():
            temp_defective_instance = defective_instance.copy(deep=True)
            subtraction = row - temp_defective_instance
            square = subtraction ** 2
            row_sum = square.apply(lambda s: s.sum(), axis=1)
            distance = row_sum ** 0.5

            temp_defective_instance["distance"] = distance
            temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
            neighbors = temp_defective_instance[1:self.z_nearest + 1]
            target_sample_instance = neighbors[-1:]
            for a in target_sample_instance.index:
                selected_pair = [index, a]
                selected_pair.sort()
                total_pair.append(selected_pair)
        total_pair_tuple = [tuple(l) for l in total_pair]
        result = Counter(total_pair_tuple)
        result_number = len(result)
        result_keys = result.keys()
        result_values = result.values()
        for f in range(result_number):
            current_pair = list(result_keys)[f]
            row1_index = current_pair[0]
            row2_index = current_pair[1]
            row1 = defective_instance.loc[row1_index]
            row2 = defective_instance.loc[row2_index]
            generated_num = list(result_values)[f]
            generated_instances = np.linspace(row1, row2, generated_num + 2)
            generated_instances = generated_instances[1:-1]
            generated_instances = generated_instances.tolist()
            for w in generated_instances:
                generated_dataset.append(w)
        final_generated_dataset = pd.DataFrame(generated_dataset)
        # final_generated_dataset = final_generated_dataset.rename(
        #     columns={0:'cosine_sim', 1:'Euclidean_dist', 2:'pearson', 3:'manhattanDisSim', 4:'cosine_sim1',
        #      5:'Euclidean_dist1', 6:'lcsSim', 7:'diceSim', 8:'editSim', 9:'levenSim',10: 'jaroSim',11: 'jaroWinklerSim',
        #     12:'re1',13: 're2',14: 're3',15: 're4', 16:'re5', 17:'re6', 18:'re7',19: 're8', 20:'re9', 21:'re10', 22:'re11',
        #     23:'javaclassnum', 24:'fieldsprob',25: 'fieldsavg', 26:'fieldssum',27: 'label'}
        #
        # )
        final_generated_dataset = final_generated_dataset.rename(
            columns={0: 'cosine_sim1', 1: 'Euclidean_dist1', 2: 'lcsSim', 3: 'diceSim', 4: 'editSim', 5: 'levenSim',
                     6: 'jaroSim', 7: 'jaroWinklerSim',
                     8: 're1', 9: 're2', 10: 're3', 11: 're4', 12: 're5', 13: 're6', 14: 're7', 15: 're8', 16: 're9',
                     17: 're10', 18: 're11',
                     19: 'javaclassnum', 20: 'fieldsprob', 21: 'fieldsavg', 22: 'fieldssum', 23: 'label'}

            )
        result = pd.concat([clean_instance, defective_instance, final_generated_dataset])
        return result