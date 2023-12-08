""" [Chapter 2 : Transformers 라이브러리 사용하기]
1) pipeline 내부 실행 과정
2) models
"""


"""
1) Pipeline 내부 실행 과정

<AutoTokenizer>
- pipeline()는 전처리(preprocessing, ex. tokenizer)->모델로 입력 전달->후처리(postprocessing) 이 3단계를 한 번에 처리
- 모든 전처리(preprocessing)는 모델이 사전 학습(pretraining)될 때와 정확히 동일한 방식으로 수행되어야 하므로 먼저 Model Hub에서 해당 정보를 다운로드해야 함
- 이를 위해 AutoTokenizer 클래스와 from_pretrained() 메서드를 사용함
- 모델의 체크포인트(checkpoint) 이름을 사용하여 모델의 토크나이저(tokenizer)와 연결된 데이터를 자동으로 가져와 캐시함
- 코드를 처음 실행할 때만 해당 정보가 다운로드됨

<AutoModel>
- 토크나이저와 동일한 방식으로 pretrained model을 다운로드 할 수 있음
- AutoTokenizer 클래스와 마찬가지로 from_pretrained() 메서드가 포함된 AutoModel 클래스를 제공함
- 출력 벡터는 세 가지의 차원으로 구성됨 -> 1.batch size, 2.sequence length, 3.hidden size
"""
from transformers import pipeline

classifier = pipeline("sentiment-analysis", device=0)
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

# pipeline 뜯어보기

# <Tokenizer>
# 단일 문장 또는 다중 문장 리스트를 토크나이저 함수로 전달할 수 있을 뿐만 아니라 출력 텐서 유형을 지정할 수 있음
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")      # input_ids, attention_mask 포함
print(inputs)

# <Model>
# 1) *Model (hidden states를 리턴)
# 2) *ForCausalLM
# 3) *ForMaskedLM
# 4) *ForMultipleChoice
# 5) *ForQuestionAnswering
# 6) *ForSequenceClassification
# 7) *ForTokenClassification
# 8) and others🤗
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)                   # output shape : (2, 16, 768)
print(outputs.last_hidden_state.shape)

# sequence classification head가 포함되어 있는 모델링 방법 -> 즉 모델 끝단에 classifier가 연결되어 있는 모델 만드는 방법
# 두 개의 문장과 두 개의 레이블만 있기 때문에, 모델에서 얻은 결과의 모양(shape)은 2 x 2
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)                   # output shape : (2, 2)
print(outputs.logits.shape)

# 출력 후처리하기
print(outputs.logits)       # 위에서 도출된 이 값은 마지막 계층에서 출력된 정규화되지 않은 원시 점수인 'logits'임

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)       # softmax 추가
print(predictions)

print(model.config.id2label)       # 각 위치에 해당하는 레이블을 가져오기 위해, model.config의 id2label 속성값을 확인


"""
2) models(모델)
- Transformer model 생성하기
- BERT 모델을 초기화하기 위해 가장 먼저 해야 할 일은 설정(configuration) 객체를 로드하는 것
- config를 불러와 모델을 생성하면 가중치와 편향이 모두 초기화되어 있음
- from_pretrained() 메서드를 사용하여 사전학습된 모델을 불러올 수 있음
- 아래 예시에서는 bert-base-cased 식별자를 통해 사전 학습된 모델을 로드하였음(BERT 개발자가 직접 학습한 모델 체크포인트)
- 자동으로 가중치가 다운로드되고 캐시되어 캐시 폴더에 저장됨(default address : ~/.cache/huggingface/hub)
- 저장 메서드 : save_pretrained() -> config.json : 모델 구조 관련 속성들, pytorch_model.bin : 가중치 저장
"""
from transformers import BertConfig, BertModel

# config(설정)을 만듭니다. -> weight and bias 모두 초기화
config = BertConfig()

# 해당 config에서 모델을 생성합니다.
model = BertModel(config)

print(config)       # hidden_size : hidden_states vector size, num_hidden_layers : transformer model의 layers 수

# 사전학습된 Bert model 생성
model = BertModel.from_pretrained("bert-base-cased")

# 모델 저장
model.save_pretrained("./bertmodel")

# 추론
sequences = ["Hello!", "Cool.", "Nice!"]

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

import torch
model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)