""" [Chapter 1 : pipeline()]
- pipeline 종류
1) feature-extraction (텍스트에 대한 벡터 표현 제공)
2) fill-mask
3) ner (named entity recognition, 개체명 인식)
4) question-answering
5) sentiment-analysis
6) summarization
7) text-generation
8) translation
9) zero-shot-classification
"""


"""
[1. sentiment-analysis]
- 텍스트 감정 분류
- 사전 훈련 모델을 불러와서 임의의 입력을 분류할 수 있음
"""
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")

# 2개 이상의 문장을 넣을 수 있음
classifier(["I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!"])


"""
[2. zero-shot classification]
- 분류에 사용할 label을 직접 설정하고 pre-trained model을 불러와 label을 예측할 수 있음
- 모델 용량 : 1.63GB
"""
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)


"""
[3. text-generation]
- 특정 프롬프트를 제공하면, 모델이 나머지 텍스틀 생성해줌(자동 완성)
- 결과는 그때그때 랜덤하게 달라짐
- generator 객체에 num_return_sequences 인자(argument)를 지정하여 생성되는 시퀀스의 개수와 max_length로 출력 텍스트의 총 길이를 제어할 수 있음
- 파이프라인에서 허브의 모든 모델을 사용할 수 있음
"""
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")

# 허브에 있는 다른 모델 불러와서 사용하기
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")    # distilgpt2 모델을 로드한다.
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)


"""
[4. mask filling]
- 주어진 텍스트의 공백을 채우는 것
- top_k argument는 출력할 공백 채우기 종류의 개수를 지정
- 즉 top_k=2이면, 2개의 후보와 각 후보들을 넣은 sequence들이 출력됨
"""
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)


"""
[5. Named entity recognition(개체명 인식)]
- 아래 예시에서 "Sylvain"이 사람(PER)이고 "Hugging Face"가 조직(ORG)이며 "Brooklyn"이 위치(LOC)임을 올바르게 식별
- grouped_entities=True 옵션을 전달하면 파이프라인이 동일한 엔티티에 해당하는 문장의 토큰(or 단어)을 그룹화함
"""
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
