import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import time

# 학습 데이터와 검증 데이터를 각각 불러오기 (파일 경로를 지정하세요)
train_df = pd.read_csv('../data/train.csv', header=None)  # train.csv 경로 입력
valid_df = pd.read_csv('../data/valid.csv', header=None)  # valid.csv 경로 입력

# 열 이름 설정
train_df.columns = ['sentence', 'label']
valid_df.columns = ['sentence', 'label']

# Hugging Face Dataset으로 변환
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

# BERT Tokenizer 로딩
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 텍스트 데이터를 BERT 입력 형식에 맞게 토큰화하는 함수 정의
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

# Tokenize train, valid dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

# BERT 모델 로딩 (이진 분류용)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Trainer를 위한 인수 설정
training_args = TrainingArguments(
    output_dir='./results',           # 결과 디렉토리
    num_train_epochs=3,               # 학습 에폭 수
    per_device_train_batch_size=8,    # 학습 배치 크기
    per_device_eval_batch_size=8,     # 평가 배치 크기
    warmup_steps=500,                 # learning rate warmup 단계
    weight_decay=0.01,                # 가중치 감소
    logging_dir='./logs',             # 로그 디렉토리
)

# Trainer 설정
trainer = Trainer(
    model=model,                         # 모델
    args=training_args,                  # 학습 인수
    train_dataset=train_dataset,         # 학습 데이터셋
    eval_dataset=valid_dataset           # 검증 데이터셋
)

start_time = time.time()
# 학습 시작
trainer.train()
end_time = time.time()

elapsed_time = end_time - start_time
print(f"학습 시간: {elapsed_time:.2f} 초")

# 모델 평가
eval_results = trainer.evaluate()
print(eval_results)


