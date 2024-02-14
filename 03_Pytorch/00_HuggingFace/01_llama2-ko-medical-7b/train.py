# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q datasets
# !pip install -q tensorboard


from datasets import load_dataset

data = load_dataset("squarelike/ko_medical_chat")
data = data.map(
    lambda x: {
        'text': "\n".join([f"{'환자' if line['from']=='client' else '의사'}: {line['value']}{'</끝>' if line['from']!='client' else ''}" for line in x['conversations']])
      }
)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_id = "squarelike/llama2-ko-medical-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

data = data.map(lambda samples: tokenizer(samples["text"], truncation=True, max_length=2048), batched=True)


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


tokenNum_ai = 33687     # "의사"
tokenNum_human = 35604   # "환자"
tokenNum_com = 29901        # ":"


import transformers
from transformers import Trainer
import numpy as np

class maskTrainer(Trainer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def compute_loss(self, model, inputs, return_outputs=False):
    for x in range(len(inputs['labels'])):
      # print(tokenizer.decode(inputs['labels'][x]))

      maskindex1 = (inputs['labels'][x]==tokenNum_human).nonzero()[:, 0].cpu()
      temp = 0
      for i, index in enumerate(maskindex1):
        if (inputs['labels'][x][index+1] != tokenNum_com):
          maskindex1 = np.delete(maskindex1, i-temp)
          temp += 1

      maskindex2 = (inputs['labels'][x]==tokenNum_ai).nonzero()[:, 0].cpu()
      temp = 0
      for i, index in enumerate(maskindex2):
        if (inputs['labels'][x][index+1] != tokenNum_com):
          maskindex2 = np.delete(maskindex2, i-temp)
          temp += 1

      for i in range(len(maskindex1)):
        ai_index = -1
        for num in maskindex2:
          if (maskindex1[i] < num):
            ai_index = num
            break
        if (ai_index == -1):
          inputs['labels'][x][maskindex1[i]+2:] = -100
        else:
          inputs['labels'][x][maskindex1[i]+2:ai_index+2] = -100
    # print(inputs['labels'][x])

    outputs = model(**inputs)

    loss = outputs['loss']

    return (loss,outputs) if return_outputs else loss


# import transformers

# # needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = maskTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # warmup_steps=200,
        # max_steps=3000, ## 초소형만 학습: 10 step = 20개 샘플만 학습.
        fp16=True,
        output_dir="outputs",
        save_total_limit=2,
        logging_steps=50,
        report_to=["tensorboard"],
        num_train_epochs=5,
        learning_rate=3e-4,
        # resume_from_checkpoint="./outputs/checkpoint-9500",
        # resume_from_checkpoint=True,
        lr_scheduler_type= "cosine",
        # optim="paged_adamw_8bit"
    
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


model.eval()
model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

model.save_pretrained("./saved/doctor/7B/try2_5epoch")


from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

stop_words = ["</끝>"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

prompt = """아래는 전문적인 의사와 환자의 진료 기록이다.

환자: 갑자기 무릎이 아파요.
의사: 언제부터 그런 증상이 있었나요?</끝>
환자: 조금 전 부터에요. 전에 이 부위가 골절된 적이 있었는데. 지금은 전부 치료되었거든요. 왜 또 아플까요?
의사: 무릎을 움직일 때 통증이 심해지나요?</끝>
환자: 네. 무릎에 힘을 주면 깨질듯이 아파요.
의사: 무릎을 움직이는 것은 가능한가요?</끝>
환자: 음...못 움직입니다. 너무 아파요
의사:"""

tokenizer.decode(model.generate(
    **tokenizer(
        prompt, 
        return_tensors='pt', 
        return_token_type_ids=False
    ),
    max_new_tokens=500,
    temperature=0.2,
    no_repeat_ngram_size=10,
    early_stopping=True,
    eos_token_id=2,
    stopping_criteria=stopping_criteria
)[0]).replace(prompt+" ", "")