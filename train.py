import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import time

train_df = pd.read_csv('../data/train.csv', header=None)  # train.csv 경로 입력
valid_df = pd.read_csv('../data/valid.csv', header=None)  # valid.csv 경로 입력

train_df.columns = ['sentence', 'label']
valid_df.columns = ['sentence', 'label']

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

# Tokenize train, valid dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',           
    num_train_epochs=3,               
    per_device_train_batch_size=8,    
    per_device_eval_batch_size=8,     
    warmup_steps=500,                 
    weight_decay=0.01,                
    logging_dir='./logs',             
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=valid_dataset           
)

start_time = time.time()
trainer.train()
end_time = time.time()

elapsed_time = end_time - start_time
print(f"학습 시간: {elapsed_time:.2f} 초")

eval_results = trainer.evaluate()
print(eval_results)


