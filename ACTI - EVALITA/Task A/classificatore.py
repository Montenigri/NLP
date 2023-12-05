import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import time
from torch.optim import Adam
from torch import nn
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.utils import shuffle
import nltk
import matplotlib.pyplot as plt
import emoji


dataset = pd.read_csv('Dataset/subtaskA_train.csv',  header=0, names=['id', 'text', 'label'])


def process_data_augmentation(dataset):
    typeA=['[Sinonimi]:','[Riassunto]:','[Tono diverso]:','[Parole chiave sostituite]:','[Domanda diretta]:']
    typeB=['Riscrivi il testo originale utilizzando sinonimi per esprimere lo stesso significato in italiano:','Fornisci un riassunto del testo originale in italiano:','Riscrivi il testo originale con un tono diverso, mantenendo però il medesimo significato in italiano:', 'Sostituisci alcune parole chiave nel testo originale in italiano:','Trasforma il testo originale in una domanda diretta in italiano:']
    processed_dataset = dataset.copy()
    processed_dataset = processed_dataset.drop(columns=['augmented_text'])
    augmented_dataset = pd.DataFrame(columns=['text', 'label'])
    #dataset['augmented_text'] = dataset['augmented_text'].apply(lambda x: x.replace('\r', ' ').replace('\n', ' '))

    for i in tqdm(range(len(dataset))):
        augmented = dataset.loc[i,'augmented_text']
        augmented = augmented.split('\n')

        for statement in augmented:
            for start in typeA:
                if start in statement:
                    augmented_dataset.loc[0,'text'] = statement.replace(start, '')
                    augmented_dataset.loc[0,'label'] = dataset.loc[i,'label']
                    processed_dataset = pd.concat([processed_dataset,augmented_dataset],ignore_index=True)
            for start in typeB:
                if start in statement:
                    augmented_dataset.loc[0,'text'] = statement.replace(start, '')
                    augmented_dataset.loc[0,'label'] = dataset.loc[i,'label']
                    processed_dataset = pd.concat([processed_dataset,augmented_dataset],ignore_index=True)
                    
    processed_dataset['text'] = processed_dataset['text'].apply(lambda x: x.replace('\r', ' ').replace('\n', ' '))

    return processed_dataset


augmented_dataset = pd.read_csv('Dataset/augmented_dataset1.csv',  header=0, names=['text', 'label','augmented_text'])
print("augmented dataset readed")
augmented_dataset = process_data_augmentation(augmented_dataset)
print("augmented dataset preprocessed")
augmented_dataset['char_count'] = augmented_dataset['text'].str.len()
print("char count done")
def add_emoji(dataset):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
    augmented_dataset = dataset.copy()
    augmented_dataset['emoji'] = ''
    augmented_dataset['numero_emoji'] = 0

    for i in tqdm(range(len(dataset)), desc= "Looking for emoji"):
        text = dataset.loc[i,'text']
        data = emoji.demojize(text, language='it').split()
        temp = []
        for i in range(len(data)):
            if data[i] not in text:
                temp.append(data[i])

        for i in range(len(temp)):
            k = temp[i].split(':')
            if len(k) > 1:
                temp[i] = k[1].replace('_',' ')

        numero_emoji = len(temp)
        emoji_found = temp   
        
        augmented_dataset.at[i,'emoji'] = emoji_found
        augmented_dataset.at[i,'numero_emoji'] = numero_emoji
    emoji_tokenized = []
    emoji_attention_mask = []
    for i in tqdm(range(len(augmented_dataset)), desc= "tokenizing emoji"):
        data_to_tokenize = augmented_dataset['emoji']
        data_tokenized = []
        for i in range(len(data_to_tokenize)):
            if len(data_to_tokenize[i])>0:
                data_tokenized.append(tokenizer(data_to_tokenize[i], add_special_tokens=True,return_tensors='pt', padding='max_length', max_length = 32, truncation=True))
            else:
                data_tokenized.append(tokenizer(' ', add_special_tokens=True,return_tensors='pt', padding='max_length', max_length = 32, truncation=True))
        
        for i in data_tokenized:
            emoji_tokenized.append(i['input_ids'])
            emoji_attention_mask.append(i['attention_mask'])
    
    augmented_dataset['emoji_tokenized'] = emoji_tokenized
    augmented_dataset['emoji_attention_mask'] = emoji_attention_mask

    
    return augmented_dataset

augmented_dataset = add_emoji(augmented_dataset)


def add_hashtag(dataset):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
    augmented_dataset = dataset.copy()
    augmented_dataset['hashtag'] = ''

    data = []
    for i in tqdm(range(len(dataset)), desc= "Looking for hashtag"):
        text = dataset.loc[i,'text']
        dato = [k for k in text.split() if k.startswith("#")]
        data.append(dato)
        
    augmented_dataset['hashtag'] = data
    data_tokenized=[]
    hashtag_tokenized = []
    hashtag_attention_mask = []
    for i in tqdm(range(len(augmented_dataset)), desc= "tokenizing hashtag"):
       
        for i in range(len(data)):
            if len(data[i])>0:
                data[i] = data[i].replace('#','')
                data_tokenized.append(tokenizer(data[i], add_special_tokens=True,return_tensors='pt', padding='max_length', max_length = 32, truncation=True))
            else:
                data_tokenized.append(tokenizer(' ', add_special_tokens=True,return_tensors='pt', padding='max_length', max_length = 32, truncation=True))
        
        for i in data_tokenized:
            hashtag_tokenized.append(i['input_ids'])
            hashtag_attention_mask.append(i['attention_mask'])

    augmented_dataset['emoji_tokenized'] = hashtag_tokenized
    augmented_dataset['emoji_attention_mask'] = hashtag_attention_mask

    return augmented_dataset

augmented_dataset = add_hashtag(augmented_dataset)


augmented_dataset.to_csv('Dataset/augmented_dataset_with_added_cols.csv', index=False)
print("augmented dataset saved")