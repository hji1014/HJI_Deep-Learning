from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

# =================================================================================================================== #

from transformers import CLIPConfig, CLIPModel

# Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
configuration = CLIPConfig()

# Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
model = CLIPModel(configuration)

# Accessing the model configuration
configuration = model.config

# We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
from transformers import CLIPTextConfig, CLIPVisionConfig

# Initializing a CLIPText and CLIPVision configuration
config_text = CLIPTextConfig()
config_vision = CLIPVisionConfig()

config = CLIPConfig.from_text_vision_configs(config_text, config_vision)

# =================================================================================================================== #

from transformers import CLIPTextConfig, CLIPTextModel

# Initializing a CLIPTextConfig with openai/clip-vit-base-patch32 style configuration
configuration = CLIPTextConfig()

# Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
model = CLIPTextModel(configuration)

# Accessing the model configuration
configuration = model.config

# =================================================================================================================== #

from transformers import CLIPVisionConfig, CLIPVisionModel

# Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
configuration = CLIPVisionConfig()

# Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
model = CLIPVisionModel(configuration)

# Accessing the model configuration
configuration = model.config

# =================================================================================================================== #

from transformers import CLIPTokenizer, CLIPTokenizerFast, AutoTokenizer

tokenizor = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")   # 얘는 vocab이 따로 안 들어 있음

tokenizor2 = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
voca2 = tokenizor2.vocab

tokenizor3 = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
voca3 = tokenizor3.vocab
print(voca3['hello'])
voca3_reverse = {v:k for k, v in voca3.items()}     # 딕셔너리의 key와 value를 바꿔서 저장하는 방법
print(voca3_reverse[12887])

print(tokenizor3(["hello", "my name is junil"]))
print(tokenizor2(["There is a sevier cancer.", "there is a severe Cancer."]))
print(tokenizor2("severe"))
print(tokenizor2("hello"))
print(tokenizor2("hello, my name is junil. I'm 30-years-old. I love you."))
# =================================================================================================================== #
# =================================================================================================================== #
# =================================================================================================================== #
# =================================================================================================================== #
# =================================================================================================================== #