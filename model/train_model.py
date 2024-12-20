import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import pickle

# Training data set
df_train = pd.read_csv('./data/train.csv', on_bad_lines='skip')
df_train.columns = ['ID', 'Label', 'Statement', 'Subject(s)', 'Speaker', 'Speaker Title', 'State Info', 'Party Affiliation', 'Barely True Counts', 'False Counts', 'Half True Counts', 'Mostly True Counts', 'Pants on Fire Counts', 'Context']

df_train['Statement'] = df_train['Statement'].fillna('')  # Fill missing text data
df_train = df_train[df_train['Label'].isin(['false', 'true'])]
df_train['Label'] = df_train['Label'].apply(lambda x: 1 if x == 'false' else 0)
df_train = df_train.dropna(subset=['Label', 'Statement'])
print(df_train['Label'].value_counts())

X_train = df_train['Statement']
y_train = df_train['Label']

# Testing data set
df_test = pd.read_csv('./data/test.csv', on_bad_lines='skip')
df_test.columns = ['ID', 'Label', 'Statement', 'Subject(s)', 'Speaker', 'Speaker Title', 'State Info', 'Party Affiliation', 'Barely True Counts', 'False Counts', 'Half True Counts', 'Mostly True Counts', 'Pants on Fire Counts', 'Context']

df_test['Statement'] = df_test['Statement'].fillna('')  # Fill missing text data
df_test = df_test[df_test['Label'].isin(['false', 'true'])]
df_test['Label'] = df_test['Label'].apply(lambda x: 1 if x == 'false' else 0)
df_test = df_test.dropna(subset=['Label', 'Statement'])
print(df_test['Label'].value_counts())


X_test = df_test['Statement']
y_test = df_test['Label']

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            self.texts[item], 
            add_special_tokens=True, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

# Parameters
MAX_LEN = 128
BATCH_SIZE = 8

# Create datasets for training and testing
train_dataset = NewsDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
test_dataset = NewsDataset(X_test.tolist(), y_test.tolist(), tokenizer, MAX_LEN)

# Load pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the Trainer
training_args = TrainingArguments(
    output_dir='../model',          # Output directory
    num_train_epochs=3,            # Number of training epochs
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,              # Number of warmup steps
    weight_decay=0.01,             # Strength of weight decay
    logging_dir='../logs',         # Directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # The model to train
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=test_dataset            # Evaluation dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('model/fake_news_bert_model')
tokenizer.save_pretrained('model/fake_news_bert_tokenizer')