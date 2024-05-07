
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer

# 加载数据集
ds = Dataset.load_from_disk("data/alpaca_data_zh")
print(ds[0])

# {'output': '以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。', 
# 'input': '', 
# 'instruction': '保持健康的三个提示。'}

# 数据处理
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")

def process_func(example):
    """
    输出结果是字典：{"input_ids":[], "attention_mask":[], "labels":[]}
    """
    MAX_LENGTH = 256
    # 对instruction和input进行格式化处理，并拼在一起进行分词
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    # 对output末尾加上结束符，并做分词处理
    response = tokenizer(example["output"] + tokenizer.eos_token)
    # input_ids需要instruction和input和output都加在一起，这样可以做多轮对话
    input_ids = instruction["input_ids"] + response["input_ids"]
    # attention_mask也需要instruction和input和output都加在一起，这样可以做多轮对话
    attention_mask = instruction['attention_mask'] + response["attention_mask"]
    # 注意labels，如果想在训练的时候，输入不参与计算损失，就需要将instruction和input那段的值设置为-100
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    # 这里进行截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)


# 创建模型
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")


"""微调方法1: BitFit"""
## 就是只更新模型参数中bias部分
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
# 训练
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    logging_steps=10,
    num_train_epochs=1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()          
# 推理
model = model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: ", return_tensor="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length = 128, do_sample = True)[0], skip_special_tokens = True)

from transformers import pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: "
pipe(ipt, max_length = 256, do_sample = True)
        
        
"""微调方法2: prompt tuning"""
## 就是在input的embedding层拼接上一个prompt的embedding，只更新这个prompt的embedding
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
"""2.1 soft prompt"""
# 只需要指定prompt的长度
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, # 指定任务类型
    num_virtual_tokens=10, # 在input前加上token长度为10的prompt
)
model = get_peft_model(model,config)
# peft提供了现成的方法来查看哪些参数可训练，哪些不一样
# model.print_trainable_parameters()
"""2.2 hard prompt"""
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, # 指定任务类型
    prompt_tuning_init=PromptTuningInit.TEXT,  # 使用TEXT方法自己指定prompt
    prompt_tuning_init_text="下面是一段人与机器人的对话。", # 指定prompt
    num_virtual_tokens=len(tokenizer("下面是一段人与机器人的对话。")["input_ids"]), # prompt的长度，要分词处理
    tokenizer_name_or_path="Langboat/bloom-1b4-zh"  # 指定分词器
)
model = get_peft_model(model,config)
# model.print_trainable_parameters()
     
# 训练
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    logging_steps=10,
    num_train_epochs=1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

# 加载训练好的peft模型
from peft import PeftModel
peft_model = PeftModel.from_pretrained(
    model = model,  # 这里的model是前面加载的model，就是model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")
    model_id="./chatbot/checkpoint-500/"  # 指定训练后的peft_model的路径
    )
# 推理
model = peft_model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: ", return_tensor="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length = 128, do_sample = True)[0], skip_special_tokens = True)

from transformers import pipeline
pipe = pipeline("text-generation", model=peft_model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: "
pipe(ipt, max_length = 256, do_sample = True)




"""微调方法3: P-Tuning"""
# 在prompt-tuning的基础上，使用prompt-encoder（prompt-embedding层+LSTM/MLP）代替prompt-embedding层
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType
config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM, # 指定是MLP还是LSTM
    encoder_hidden_size=1024,
    encoder_dropout=0.5,
    encoder_num_layers=4, # 以上三个参数是调节encoder部分LSTM或者MLP的参数
)
model = get_peft_model(model, config)
# model.print_trainable_parameters()
# 训练
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    logging_steps=10,
    num_train_epochs=1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

# 加载训练好的peft模型
from peft import PeftModel
peft_model = PeftModel.from_pretrained(
    model = model,  # 这里的model是前面加载的model，就是model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")
    model_id="./chatbot/checkpoint-500/"  # 指定训练后的peft_model的路径
    )
# 推理
model = peft_model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: ", return_tensor="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length = 128, do_sample = True)[0], skip_special_tokens = True)

from transformers import pipeline
pipe = pipeline("text-generation", model=peft_model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: "
pipe(ipt, max_length = 256, do_sample = True)



"""微调方法4: Prefix-Tuning"""
from peft import PrefixTuningConfig, get_peft_model, TaskType
config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    prefix_projection=False
)
model = get_peft_model(model, config)
# model.print_trainable_parameters()
# 训练
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    logging_steps=10,
    num_train_epochs=1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

# 加载训练好的peft模型
from peft import PeftModel
peft_model = PeftModel.from_pretrained(
    model = model,  # 这里的model是前面加载的model，就是model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")
    model_id="./chatbot/checkpoint-500/"  # 指定训练后的peft_model的路径
    )
# 推理
model = peft_model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: ", return_tensor="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length = 128, do_sample = True)[0], skip_special_tokens = True)

from transformers import pipeline
pipe = pipeline("text-generation", model=peft_model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: "
pipe(ipt, max_length = 256, do_sample = True)





"""微调方法5: Lora"""
from peft import LoraConfig, get_peft_model, TaskType
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32, # 缩放，真实缩放值是lora_alpha/r
    target_modules=["query_key_value"], # 可以自己添加想要加Lora的层
    modules_to_save=["word_embeddings"] # 除了Lora层需要训练，还希望哪些参数训练
)
model = get_peft_model(model, config)
# model.print_trainable_parameters()
# 训练
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    logging_steps=10,
    num_train_epochs=1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

# 加载训练好的peft模型
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
p_model = PeftModel.from_pretrained(
    model = model,
    model_id="./chatbot/checkpoint-500/"  # 指定训练后的peft_model的路径
    )
# 推理
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: ", return_tensor="pt")
tokenizer.decode(p_model.generate(**ipt, max_length = 128, do_sample = True)[0], skip_special_tokens = True)

# 模型合并，将peft_model和基座的model合并
merge_model = p_model.merge_and_unload()

# 模型保存
merge_model.save_pretrained("./chabot/merged_Lora_Model")

from transformers import pipeline
pipe = pipeline("text-generation", model=merge_model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: "
pipe(ipt, max_length = 256, do_sample = True)




"""微调方法6:  IA3 """
# 通过抑制和放大激活值，通过可学习的向量对激活值进行抑制或放大，具体对K、V、FFN三部分的值进行调整
from peft import IA3Config, TaskType, get_peft_model
config = IA3Config(
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, config)
# model.print_trainable_parameters()
# 训练
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    logging_steps=10,
    num_train_epochs=1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

# 加载训练好的peft模型
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
p_model = PeftModel.from_pretrained(
    model = model,
    model_id="./chatbot/checkpoint-500/"  # 指定训练后的peft_model的路径
    )
# 推理
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: ", return_tensor="pt")
tokenizer.decode(p_model.generate(**ipt, max_length = 128, do_sample = True)[0], skip_special_tokens = True)

# 模型合并，将peft_model和基座的model合并
merge_model = p_model.merge_and_unload()

# 模型保存
merge_model.save_pretrained("./chabot/merged_Lora_Model")

from transformers import pipeline
pipe = pipeline("text-generation", model=merge_model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？","").strip() + "\n\nAssistant: "
pipe(ipt, max_length = 256, do_sample = True)
