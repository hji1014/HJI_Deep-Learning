""" [Chapter 2 : Transformers 라이브러리 사용하기]
1) pipeline 내부 실행 과정
2) models
3) tokenizer
4) handling multiple sequences(다중 시퀀스 처리)
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


"""
3) tokenizer(토크나이저)
- 단어 기반 토큰화 (Word-based Tokenization)
    - split on spaces
    - split on punctuation
    - 모든 단어를 감당할려면 엄청나게 많은 메모리가 소비됨
    - 중요 단어만 추가하면 out-of-vocabulary (OOV, unknown) 문제에 직면하게 됨
- 문자 기반 토큰화 (Character-based Tokenization)
- 하위 단어 토큰화 (Subword Tokenization)
    - 위의 두 가지 방법을 모두 종합한 방법
    - ex) annoyingly -> annoying + ly
- 세부 기법들
    - Byte-level BPE (GPT-2에 사용됨)
    - WordPiece (BERT에 사용됨)
    - SentencePiece, Unigram (몇몇 다국어 모델에 사용됨)
    
- 토크나이저 로딩 및 저장 : '모델 로드 및 저장'과 마찬가지로 from_pretrained()와 save_pretrained() 메서드를 그대로 사용

- Encoding : 토큰화 + 입력 식별자(input IDs)로의 변환 -> 2단계
- Decoding : 다시 text로 변환하는 인코딩 반대 과정
"""

# 토크나이저 로딩(1)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer("Using a Transformer network is simple")

# 토크나이저 로딩(2)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer("Using a Transformer network is simple")

# 토크나이저 저장
tokenizer.save_pretrained("./bert_tok")

# 토큰화 작업(인코딩)
sequence = "Using a Transformer network is simple"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokens = tokenizer.tokenize(sequence)
print(tokens)

# 토큰을 input IDs로 변환(인코딩)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# 디코딩
decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
print(decoded_string)

"""
4) handling multiple sequences(다중 시퀀스 처리)
- model은 입력의 batch 형태를 요구함 -> 1개 데이터 shape : (1, 14), 2개 데이터 shape : (2, 14)
- 모든 시퀀스의 길이를 동일하게 패딩해줘야 함. -> 패딩 토큰 식별자(ID)는 tokenizer.pad_token_id에 저장되어 있음
    -> 대부분의 모델은 sequence length를 512개 또는 1024개까지 처리할 수 있음
    -> 더 긴 sequence를 처리하기 위해서는 다른 모델을 쓰던가, sequence를 잘라서 넣어야 함(sequence = sequence[:max_sequence_length])
    -> 절단하는 것을 truncation이라고 함
"""

# 에러 코드 -> (14,) 크기의 텍스트 하나만 달랑 넣었을 때 -> 에러 발생
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)

model(input_ids)        # This line will fail

tokenized_inputs = tokenizer(sequence, return_tensors="pt")         # 이렇게 해야 함
print(tokenized_inputs["input_ids"])

# 제대로 입력 넣는 방법
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])         # 시퀀스 1개
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

batched_ids = [ids, ids]                # 시퀀스 2개
input_ids = torch.tensor(batched_ids)

output = model(input_ids)
print("Logits:", output.logits)

# 입력을 padding 하기(padding 전) -> 차원이 맞지 않아 tensor로 변환 안됨
batched_ids = [
    [200, 200, 200],
    [200, 200],
]

# 입력을 padding 하기(padding token : 100)
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]

# 패딩한 시퀀스를 모델에 넣어보기
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)          # sequence2와 batched_ids의 두 번째 데이터로부터 추론된 결과가 다름 -> pad 토큰의 영향(with attention layers) -> attention mask를 사용하여 패딩 토큰을 무시하게 만들어야 함

batch_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]
outputs = model(torch.tensor(batch_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)       # attention mas를 사용하면 sequence2와 같은 결과가 추론됨

