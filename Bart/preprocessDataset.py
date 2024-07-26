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
        
        static_data_per_column = column_input + ' ; ' + row_input
        text = ' '.join(row['title'].strip().split()) + '<s>' + static_data_per_column
        summary = row['first_caption'].strip()
        each_json = {'text': text, 'summary': summary}
        file.append(each_json)
        
    return random.shuffle(file)

datasets_multi = load_dataset('csv', data_files=['../Statista_Dataset/dataset/multiColumn/metadata.csv'])
datasets_basic = load_dataset('csv', data_files=['../Statista_Dataset/dataset/metadata.csv'])


data_1 = preprocessData(datasets_multi['train'], True)
data_2 = preprocessData(datasets_basic['train'], False)


data = data_1 + data_2
random.shuffle(data)

train_size = int(0.7 * len(data))
valid_size = int(0.2 * len(data))

train_file = data[:train_size]
valid_file = data[train_size : train_size + valid_size]
test_file = data[train_size + valid_size:]

with open("data/train.json", "w") as file:
    json.dump(train_file, file, indent=2)
with open("data/test.json", "w") as file:
    json.dump(test_file, file, indent=2)
with open("data/valid.json", "w") as file:
    json.dump(valid_file, file, indent=2)
print("Complete Preprocess")