from datasets import load_dataset
import pandas as pd
import json
import re
import random


def readCSV(file_path):
    df = pd.read_csv(file_path)
    return df

def cleanAxisValue(value):
    if value == '-' or value == 'nan':
        return '0'
    cleanValue = re.sub('\s', '', value)
    cleanValue = cleanValue.replace('|', '').replace(',', '').replace('%', '').replace('*', '')
    return cleanValue

def preprocessData(datasets, is_multi):
    file = []
    for row in datasets:
        print(row['dataPath'])
        if is_multi:
            df  = readCSV('../Statista_Dataset/dataset/multiColumn/' + row['dataPath'][14:])
        else:
            df  = readCSV('../Statista_Dataset/dataset/' + row['dataPath'][12:])
        column = df.columns.tolist()
        column_input = ' | '.join(cleanAxisValue(column[i]) for i in range(len(column)))
        row_input = ' ; '.join(' | '.join(cleanAxisValue(str(df.loc[index_row, col])) for col in column) for index_row in range(len(df)))
        input = column_input + ' ; ' + row_input
        
        instruction = ' '.join(row['title'].strip().split())
        output = row['first_caption'].strip()
        each_json = {'instruction': instruction, 'input': input, 'output': output}
        file.append(each_json)
        
    return random.shuffle(file)

datasets_multi = load_dataset('csv', data_files=['../Statista_Dataset/dataset/multiColumn/metadata.csv'])
datasets_basic = load_dataset('csv', data_files=['../Statista_Dataset/dataset/metadata.csv'])


data_1 = preprocessData(datasets_multi['train'], True)
data_2 = preprocessData(datasets_basic['train'], False)


data = data_1 + data_2
random.shuffle(data)

train_valid_size = int(0.9 * len(data))

train_valid_file = data[:train_valid_size]
test_file = data[train_valid_size:]

with open("data/train_valid.json", "w") as file:
    json.dump(train_valid_file, file, indent=2)
with open("data/test.json", "w") as file:
    json.dump(test_file, file, indent=2)
print("Complete Preprocess")