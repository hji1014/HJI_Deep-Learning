""" [Chapter 3 : 사전학습 모델에 대한 미세조정]
1) 데이터 처리 작업
2) Trainer API를 이용한 모델 미세조정(fine-tuning)
3) 전체 학습(full training)
"""


"""
1) 데이터 처리 작업
- 실습을 위해 MRPC (Microsoft Research Paraphrase Corpus) 데이터셋 사용
- 위 데이터셋은 5,801건의 문장 쌍으로 구성되어 있으며 각 문장 쌍의 관계가 의역(paraphrasing) 관계인지 여부를 나타내는 레이블 존재
    -> 두 문장이 동일 관계인지 여부를 나타내는 label이 존재함
- 허브(hub)에는 모델뿐만 아니라 다양한 언어로 구축된 여러 데이터셋이 있음
- 위 데이터셋은 10개의 데이터셋으로 구성된 GLUE 벤치마크 중 하나
- GLUE 벤치마크 : 10가지 텍스트 분류 작업을 통해서 기계학습 모델의 성능을 측정하기 위한 학술적 벤치마크 데이터 집합
- load_dataset 명령은 기본적으로 ~/.cache/huggingface/dataset에 데이터셋을 다운로드하고 임시저장(캐시, cache)함
- HF_HOME 환경 변수를 설정하여 캐시 폴더를 변경할 수 있습니다.
- 토크나이저(tokenizer)는 한 쌍의 시퀀스를 가져와 BERT 모델이 요구하는 입력 형태로 구성할 수 있음
"""
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "bert-base-uncased"                                                    # 2장의 예제와 동일합니다.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

batch["labels"] = torch.tensor([1, 1])                                              # 새롭게 추가된 코드입니다.

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

# 허브에서 데이터셋 로딩
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)     # train/val/test 집합이 저장된 DatasetDict 객체를 얻을 수 있음

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])             # 데이터에 접근할 수 있음

print(raw_train_dataset.features)       # 데이터타입, label 종류 등을 확인할 수 있음

# 데이터셋 전처리
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])       # 첫 번째 문장 토큰화
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])       # 두 번째 문장 토큰화

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)       # token_type_ids : sentence1, sentence2를 알려주는 리스트

tokenizer.convert_ids_to_tokens(inputs["input_ids"])        # 디코딩해보니 -> [CLS] 문장1 [SEP] 문장2 [SEP]

tokenized_dataset = tokenizer(                              # 실제로 전처리 수행 (전체 데이터를 불러서)
    raw_datasets["train"]["sentence1"],                     # RAM 낭비가 심할 수 있음
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# 동적 패딩(Dynamic Padding) -> 효율성을 위해 배치 단위로 패딩하는 방법
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

print([len(x) for x in samples["input_ids"]])       # sample들의 sequence length

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})       # padding 완료 -> 모두 동일한 67의 길이를 가짐


"""
2) Trainer API를 이용한 모델 미세 조정(fine-tuning)
- 2장과 달리, 사전 학습된 모델을 인스턴스화한 후 경고(warnings)가 출력되는 것을 알 수 있음.
    이는 BERT가 문장 쌍 분류에 대해 사전 학습되지 않았기 때문에 사전 학습된 모델의 헤드(model head)를 버리고 시퀀스 분류에 적합한 새로운 헤드를 대신 추가했기 때문.
    경고의 내용은 일부 가중치(weights)가 사용되지 않았으며(제거된 사전 학습 헤드에 해당하는 가중치) 일부 가중치가 무작위로 초기화되었음을(새로운 헤드에 대한 가중치) 나타냄
- Trainer를 정의하기 전에 먼저 수행할 단계는 Trainer가 학습 및 평가에 사용할 모든 hyperparameters를 포함하는 TrainingArguments 클래스를 정의해야 함
- Trainer 정의
"""
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 학습(Training)
from transformers import TrainingArguments                      # TrainingArguments class 정의
training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification     # 모델 정의
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer                                # Trainer 정의
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()     # 미세조정할 때  -> compute_metrics() 설정하지 않음 , evaluation_strategy="steps" or "epoch"로 설정하지 않음
                    # 따라서 training 성능(acc, F1 score 등)을 알 수 없음
# 평가
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np                                                      # logit에서 예측 label로 변환
preds = np.argmax(predictions.predictions, axis=-1)

from datasets import load_metric                                        # prediction과 label을 비교
metric = load_metric("glue", "mrpc")
print(metric.compute(predictions=preds, references=predictions.label_ids))

a = metric.compute(predictions=preds, references=predictions.label_ids)['accuracy']

# compute_metric 함수 만들어, 각 epoch가 끝날 때 metric 출력하는 Trainer 만들기
def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

