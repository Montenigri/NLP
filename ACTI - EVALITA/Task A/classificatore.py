import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import time
from torch.optim import Adam
from torch import nn
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
import nltk
import matplotlib.pyplot as plt
import emoji

pd.set_option('display.max_colwidth', None)

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

augmented_dataset = process_data_augmentation(augmented_dataset)
augmented_dataset['char_count'] = augmented_dataset['text'].str.len()



import re
def add_emoji(dataset):

    augmented_dataset = dataset.copy()
    augmented_dataset['emoji'] = ' '

    for i in tqdm(range(len(dataset)), desc= "Looking for emoji"):
        text = dataset.loc[i,'text']
        data = emoji.demojize(text, language='it')
        pattern = r":(\w+):"

        emoji_found = re.findall(pattern,data)
        emoji_found = ' '.join(emoji_found)
        augmented_dataset.at[i,'emoji'] = emoji_found

    return augmented_dataset

augmented_dataset = add_emoji(augmented_dataset)

def add_hashtag(dataset):
    augmented_dataset = dataset.copy()
    augmented_dataset['hashtag'] = ' '
    data = []

    for i in tqdm(range(len(dataset)), desc= "Looking for hashtag"):
        text = dataset.loc[i,'text']
        dato = [k for k in text.split() if k.startswith("#")]
        data.append(dato)

    augmented_dataset['hashtag'] = data
    augmented_dataset['hashtag'] = augmented_dataset['hashtag'].apply(lambda x: " ".join(x).replace('#', ''))


    return augmented_dataset

augmented_dataset = add_hashtag(augmented_dataset)


augmented_dataset = shuffle(augmented_dataset, random_state=42)

augmented_dataset.to_csv('Dataset/augmented_dataset_with_added_cols.csv', index=False)


######



hyperparameters = {
    "#_classes" : 2,
    "epochs": 10,
    "learning_rate": 1e-5,
    "batch_size": 8,
    "dropout": 0.1,
    "stopwords": False,
    "h_dim": 768,
    "patience": 5,
    "min_delta": 0.01,
    "language_model": "bert-base-multilingual-cased",
    "extra_features": 65, #32 emoji + 32 hashtag + 1 char_count
}

(x_train, x_test,char_count_train,char_count_test, hashtag_train, hashtag_test,emoji_train,emoji_test,y_train, y_test) = train_test_split(augmented_dataset['text'],augmented_dataset['char_count'], augmented_dataset['hashtag'],augmented_dataset['emoji'] ,augmented_dataset['label'], test_size=0.2, random_state=42)

(x_train, x_val,char_count_train,char_count_val, hashtag_train, hashtag_val, emoji_train, emoji_val, y_train, y_val) = train_test_split( x_train,char_count_train,hashtag_train,emoji_train, y_train, test_size=0.1, random_state=42)


nltk.download('punkt')
nltk.download('stopwords')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, char_count, emoji, hashtag, y, stopwords):


        tokens_litt = [nltk.word_tokenize(text, language='italian') for text in list(x)]

        text_clean = []

        for sentence in tqdm(tokens_litt, desc='Tokenizing text ... '):
            text_clean.append(' '.join([w for w in sentence if not w.lower() in nltk.corpus.stopwords.words("italian")]))


        self.texts = [text for text in text_clean]
        self.emoji = [e for e in emoji]
        self.hashtag = [h for h in hashtag]
        self.labels = [torch.tensor(label) for label in y]
        self.char_count = [torch.tensor(char_count) for char_count in char_count]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_char_count(self, idx):
        # Fetch a batch of inputs
        return self.char_count[idx]

    def get_batch_emoji(self, idx):
        # Fetch a batch of inputs
        return self.emoji[idx]

    def get_batch_hashtag(self, idx):
        # Fetch a batch of inputs
        return self.hashtag[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_labels = self.get_batch_labels(idx)
        batch_char_count = self.get_batch_char_count(idx)
        batch_emoji = self.get_batch_emoji(idx)
        batch_hashtag = self.get_batch_hashtag(idx)

        return batch_texts, batch_labels, batch_char_count, batch_emoji, batch_hashtag

train_dataset = Dataset(x_train, char_count_train,emoji_train,hashtag_train, y_train, hyperparameters["stopwords"])
val_dataset = Dataset(x_val,char_count_val,emoji_val,hashtag_val, y_val, hyperparameters["stopwords"])
test_dataset = Dataset(x_test, char_count_test,emoji_test,hashtag_test, y_test, hyperparameters["stopwords"])

'''
class ClassifierDeep(nn.Module):

    def __init__(self, labels, hdim, dropout, model_name,extra_features = hyperparameters['extra_features']):
        super(ClassifierDeep, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.lm_model = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hdim + extra_features, 64),
            nn.ReLU(),
            nn.Linear(64, labels),
            )

    def forward(self, input_id_text, attention_mask, char_count,emoji_text, hashtag):
        output = self.lm_model(input_id_text, attention_mask).last_hidden_state
        output = output[:,0,:]
        output = torch.cat((output, char_count.unsqueeze(-1),emoji_text,hashtag), dim=1)  # Concatena il conteggio dei caratteri
        return self.classifier(output)  
'''
class ClassifierDeep(nn.Module):

    def __init__(self, labels, hdim, dropout, model_name,extra_features = hyperparameters['extra_features']):
        super(ClassifierDeep, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.lm_model = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hdim + extra_features, 2)
            )

    def forward(self, input_id_text, attention_mask, char_count,emoji_text, hashtag):
        output = self.lm_model(input_id_text, attention_mask).last_hidden_state
        output = output[:,0,:]
        output = torch.cat((output, char_count.unsqueeze(-1),emoji_text,hashtag), dim=1)  # Concatena il conteggio dei caratteri
        return self.classifier(output)
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):

        self.patience = patience
        self.min_delta = min_delta              # valore minimo di decrescita della loss di validazione all'epoca corrente
                                                # per asserire che c'è un miglioramenti della loss
        self.counter = 0                        # contatore delle epoche di pazienza
        self.early_stop = False                 # flag di early stop
        self.min_validation_loss = torch.inf    # valore corrente ottimo della loss di validazione

    def __call__(self, validation_loss):
        # chiamata in forma funzionale dell'oggetto di classe EarlySopping
        if (validation_loss + self.min_delta) >= self.min_validation_loss:  # la loss di validazione non decresce
            self.counter += 1                                               # incrementiamo il contatore delle epoche di pazienza
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stop!")
        else:                                                               # c'è un miglioramento della loss:
            self.min_validation_loss = validation_loss                      # consideriamo la loss corrente
                                                                            # come nuova loss ottimale
            self.counter = 0                                                # e azzeriamo il contatore di pazienza



def train_loop(model, dataloader, tokenizer, loss, optimizer, device):
    model.train()

    epoch_acc = 0
    epoch_loss = 0

    for batch_texts, batch_labels, batch_char_count, emoji, hashtag in tqdm(dataloader, desc='training set'):

        optimizer.zero_grad()
        tokens = tokenizer(list(batch_texts), add_special_tokens=True,
                            return_tensors='pt', padding='max_length',
                            max_length = 400, truncation=True)
        tokens_emoji = tokenizer(list(emoji), add_special_tokens=True,
                            return_tensors='pt', padding='max_length',
                            max_length = 32, truncation=True)
        tokens_hashtag = tokenizer(list(hashtag), add_special_tokens=True,
                            return_tensors='pt', padding='max_length',
                            max_length = 32, truncation=True)
        
        input_id_texts = tokens['input_ids'].squeeze(1).to(device)
        mask_texts = tokens['attention_mask'].squeeze(1).to(device)

        input_id_emoji = tokens_emoji['input_ids'].squeeze(1).to(device)
        mask_emoji = tokens_emoji['attention_mask'].squeeze(1).to(device)

        input_id_hashtag = tokens_hashtag['input_ids'].squeeze(1).to(device)
        mask_hashtag = tokens_hashtag['attention_mask'].squeeze(1).to(device)


        batch_char_count = batch_char_count.to(device)
        batch_labels = batch_labels.to(device)
        batch_char_count = batch_char_count.to(device)

        
        output = model(input_id_texts, mask_texts,batch_char_count, input_id_emoji, input_id_hashtag)


        # la loss è una CrossEntropyLoss, al suo interno ha la logsoftmax + negative log likelihood loss
        batch_loss = loss(output, batch_labels)
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

        # per calcolare l'accuracy devo generare le predizioni applicando manualmente la logsoftmax
        softmax = nn.LogSoftmax(dim=1)
        epoch_acc += (softmax(output).argmax(dim=1) == batch_labels).sum().item()

        batch_labels = batch_labels.detach().cpu()
        input_id_texts = input_id_texts.detach().cpu()
        mask_texts = mask_texts.detach().cpu()
        output = output.detach().cpu()
        batch_char_count = batch_char_count.detach().cpu()


    return epoch_loss/len(dataloader), epoch_acc


def test_loop(model, dataloader, tokenizer, loss, device):
    model.eval()

    epoch_acc = 0
    epoch_loss = 0

    with torch.no_grad():

        for batch_texts, batch_labels, batch_char_count, emoji, hashtag in tqdm(dataloader, desc='dev set'):

            tokens = tokenizer(list(batch_texts), add_special_tokens=True,
                               return_tensors='pt', padding='max_length',
                               max_length = 512, truncation=True)
            
            tokens_emoji = tokenizer(list(emoji), add_special_tokens=True,
                            return_tensors='pt', padding='max_length',
                            max_length = 32, truncation=True)
            tokens_hashtag = tokenizer(list(hashtag), add_special_tokens=True,
                                return_tensors='pt', padding='max_length',
                                max_length = 32, truncation=True)
            
            input_id_texts = tokens['input_ids'].squeeze(1).to(device)
            mask_texts = tokens['attention_mask'].squeeze(1).to(device)

            input_id_emoji = tokens_emoji['input_ids'].squeeze(1).to(device)
            mask_emoji = tokens_emoji['attention_mask'].squeeze(1).to(device)

            input_id_hashtag = tokens_hashtag['input_ids'].squeeze(1).to(device)
            mask_hashtag = tokens_hashtag['attention_mask'].squeeze(1).to(device)

            batch_labels = batch_labels.to(device)
            batch_char_count = batch_char_count.to(device)
            output = model(input_id_texts, mask_texts,batch_char_count,input_id_emoji, input_id_hashtag)

            batch_loss = loss(output, batch_labels)
            epoch_loss += batch_loss.item()

            softmax = nn.LogSoftmax(dim=1)
            epoch_acc += (softmax(output).argmax(dim=1) == batch_labels).sum().item()

            batch_labels = batch_labels.detach().cpu()
            input_id_texts = input_id_texts.detach().cpu()
            mask_texts = mask_texts.detach().cpu()
            output = output.detach().cpu()
            batch_char_count = batch_char_count.detach().cpu()

    return epoch_loss/len(dataloader), epoch_acc

def train_test(model, epochs, optimizer, device, train_data, test_data,
               batch_size, language_model, train_loss_fn, test_loss_fn=None,
               early_stopping=None, val_data=None, scheduler=None):

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # check sulle funzioni di loss
    if test_loss_fn == None:
        test_loss_fn = train_loss_fn

    # liste dei valori di loss e accuracy epoca per epoca per il plot
    train_loss = []
    validation_loss = []
    test_loss = []

    train_acc = []
    validation_acc = []
    test_acc = []

    tokenizer = AutoTokenizer.from_pretrained(language_model)

    # Ciclo di addestramento con early stopping
    for epoch in tqdm(range(1,epochs+1)):

        epoch_train_loss, epoch_train_acc = train_loop(model,
                    train_dataloader, tokenizer, train_loss_fn, optimizer, device)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc/len(train_data))

        # validation se è presente la callback di early stopping
        if early_stopping != None:
                epoch_validate_loss, epoch_validate_acc = test_loop(model,
                                val_dataloader, tokenizer, test_loss_fn, device)
                validation_loss.append(epoch_validate_loss)
                validation_acc.append(epoch_validate_acc/len(val_data))
        # test
        epoch_test_loss, epoch_test_acc,= test_loop(model,
                                test_dataloader, tokenizer, test_loss_fn, device)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc/len(test_data))

        val_loss_str = f'Validation loss: {epoch_validate_loss:6.4f} ' if early_stopping != None else ' '
                        # ' if early_stopping != None else ' '
        val_acc_str = f'Validation accuracy: {(epoch_validate_acc/len(val_data)):6.4f} ' if early_stopping != None else ' '
                        # ' if early_stopping != None else ' '
        print(f"\nTrain loss: {epoch_train_loss:6.4f} {val_loss_str}Test loss: {epoch_test_loss:6.4f}")
                        # Test loss: {epoch_test_loss:6.4f}")
        print(f"Train accuracy: {(epoch_train_acc/len(train_data)):6.4f} {val_acc_str}Test accuracy: {(epoch_test_acc/len(test_data)):6.4f}")
                        # {val_acc_str}Test accuracy:
                        # {(epoch_test_acc/len(test_data)):6.4f}")

        # early stopping
        if early_stopping != None:
                early_stopping(epoch_validate_loss)
                if early_stopping.early_stop:
                    break

    return train_loss, validation_loss, test_loss, train_acc, validation_acc, test_acc
                        # train_acc, validation_acc, test_acc

# Acquisiamo il device su cui effettueremo il training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = ClassifierDeep(hyperparameters["#_classes"],
                    hyperparameters["h_dim"],
                    hyperparameters["dropout"],
                    hyperparameters["language_model"]).to(device)
print(model)

# Calcoliamo il numero totale dei parametri del modello
total_params = sum(p.numel() for p in model.parameters())
print(f"Numbero totale dei parametri: {total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=hyperparameters["learning_rate"])

# Creiamo la callback di early stopping da passare al nostro metodo di addestramento
early_stopping = EarlyStopping(patience=hyperparameters['patience'],
                               min_delta=hyperparameters['min_delta'])

train_loss, validation_loss,test_loss, train_acc, validation_acc, test_acc = train_test(model,
                                                # train_test(model,
                                                hyperparameters['epochs'],
                                                #50,
                                                optimizer,
                                                device,
                                                train_dataset,
                                                test_dataset,
                                                hyperparameters['batch_size'],
                                                hyperparameters['language_model'],
                                                criterion,
                                                criterion,
                                                early_stopping,
                                                val_dataset)


torch.save({
            'epoch': 10, #Epoca finale, da modificare in modo da renderla dinamica
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, "model_10_epochs")