import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# BERT 모델 및 토크나이저 로드
model_name = 'bert-base-uncased'  # 또는 다른 BERT 모델을 선택할 수 있습니다.
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 예제 데이터 생성
sentences = ["This is a positive example.", "This is a negative example."]
labels = [1, 0]  # 1: 긍정, 0: 부정

# 토큰화 및 입력 데이터 생성
tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']
labels = torch.tensor(labels)

# DataLoader로 데이터 로드
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델 훈련
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 테스트 예제
test_sentence = "I have a bad news."
test_tokenized = tokenizer(test_sentence, return_tensors='pt')
test_input_ids = test_tokenized['input_ids']
test_attention_mask = test_tokenized['attention_mask']

# 추론
with torch.no_grad():
    output = model(test_input_ids, attention_mask=test_attention_mask)

# 결과 출력
predicted_label = torch.argmax(output.logits).item()
print("Predicted Label:", predicted_label)

# 테스트 예제
test_sentence = "I have a bad news."
test_tokenized = tokenizer(test_sentence, return_tensors='pt')
test_input_ids = test_tokenized['input_ids']
test_attention_mask = test_tokenized['attention_mask']

# 추론
with torch.no_grad():
    output = model(test_input_ids, attention_mask=test_attention_mask)

# 결과 출력
predicted_label = torch.argmax(output.logits).item()
print("Predicted Label:", predicted_label)
