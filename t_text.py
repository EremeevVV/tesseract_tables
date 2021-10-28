import torch
import torchtext
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import nltk
import json
from collections import defaultdict

nltk.download('wordnet', download_dir='venv/nltk_data')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
with open('embedding/english_embedding_pt.pickle', 'rb') as f:
    embedding_dict = defaultdict(lambda: torch.zeros(50, dtype=torch.float32), pickle.load(f))


class CustomTextDataset(Dataset):
    def __init__(self, text: list[str], labels: list[int], transform_text=None, transform_label=None):
        self.labels = labels
        self.text = text
        self.transform_text = transform_text
        self.transform_label = transform_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        if self.transform_text:
            text = self.transform_text(text)
        if self.transform_label:
            label = self.transform_label(label)
        sample = {"Text": text, "Class": label}
        return sample


class JsonRecordDataset(Dataset):
    """Dataframe saved as record json load to dataset"""

    def __init__(self, json_str: str, transform_text=None, transform_label=None):
        samples = json.loads(json_str)
        self.texts = [sample['Review'] for sample in samples]
        self.labels = [sample['Rating'] for sample in samples]
        self.transofrm_text = transform_text
        self.transofrm_label = transform_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.transofrm_text:
            text = self.transofrm_text(text)
        if self.transofrm_label:
            label = self.transofrm_label(label)
        return text, label


class LongJsonDataset(IterableDataset):
    """If long json, new object in new row"""

    def __init__(self, file, transform_text=None, transform_label=None):
        super(LongJsonDataset).__init__()
        self.file = file
        self.transform_text = transform_text
        self.transform_label = transform_label

    def __iter__(self):
        with open(self.file, 'r') as f:
            for sample_line in f:
                sample = json.loads(sample_line)
                text = sample['Review']
                label = sample['Rating']
                if self.transform_text:
                    text = self.transform_text(text)
                if self.transform_label:
                    label = self.transform_label(label)
                yield {"Text": text, "Class": label}

class TextClassificationModel(torch.nn.Module):

    def __init__(self, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.fc = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        return self.fc(text)

def text_pipeline(text:str):
    # надо разобраться с видимостью обработчиков
    return [embedding_dict[lemmatizer.lemmatize(token)] for token in tokenizer(text)]

def label_pipeline(label:int) -> int:
    return label -1

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.stack(text_pipeline(_text))
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to('cpu'), text_list.to('cpu'), offsets.to('cpu')

def get_data_from_json():
    with open('data/train.json','r') as f:
        train_ds = JsonRecordDataset(f.read())
    with open('data/test.json', 'r') as f:
        test_ds = JsonRecordDataset(f.read())
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True,collate_fn=collate_batch)
    test_dl = DataLoader(test_ds, batch_size=2, shuffle=True)
    print(next(iter(train_dl))[1])
    # print(type(train_ds[2][1]))


def convert_to_train_test_jsons():
    lemmatizer = WordNetLemmatizer()
    df = pd.read_csv('data/tripadvisor_hotel_reviews.csv', header=0, encoding='UTF8', sep=',', index_col=None,
                     dtype={'Review': str, 'Rating': int})
    # print(df.groupby('Rating').count().Review.to_list())
    rat5 = df[df['Rating'] == 5].sample(4000)
    rat4 = df[df['Rating'] == 4].sample(3000)
    rat1_3 = df[df['Rating'] < 4]
    df = rat1_3.append(rat4.append(rat5, ignore_index=True), ignore_index=True)
    # tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    # df['Review'] = df['Review'].apply(lambda x: [lemmatizer.lemmatize(token) for token in tokenizer(x)])

    X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Rating'], test_size=0.33, random_state=42,
                                                        stratify=df['Rating'])
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_json(path_or_buf='data/train.json', orient='records', lines=False)
    test.to_json(path_or_buf='data/test.json', orient='records', lines=False)
    # with open('embedding/english_embedding_pt.pickle', 'rb') as f:
    #     embedding_dict = pickle.load(f)


if __name__ == '__main__':
    get_data_from_json()
