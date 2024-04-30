

# 导包
from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,Trainer,DataCollatorForLanguageModeling
from datasets import *
import evaluate
from transformers import pipeline

# 读取数据集
dataset = Dataset.load_from_disk("02-NLP Tasks/14-language_model/wiki_cn_filtered/")


# 数据处理
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-389m-zh")
def preposs_func(example):
    # 当batched=True的时候，example是Dict[str, List],List就是将所有的example的value合并成一个字典
    contents = [e + tokenizer.eos_token for e in example['completion']]
    return tokenizer(contents, max_length=64, truncation = True)
prepossed_dataset = dataset.map(preposs_func, batched = True, remove_columns=dataset.column_names)

# 划分训练集测试集
train_dataset, test_dataset = prepossed_dataset.train_test_split(test_size=0.1)


# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-389m-zh")


# 配置参数
args = TrainingArguments(
    output_dir="02-NLP Tasks/14-language_model/checkpoints/",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=32,
    evaluation_strategy="steps",
    eval_steps=10,
    logging_steps=10,
    load_best_model_at_end=True
)

# 构建评估函数
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis = -1)
    evaluate_metric = evaluate.combine([accuracy_metric, f1_metric])
    metrics = evaluate_metric.compute(predictions=predictions,references=labels)
    return metrics
    


# 创建训练器
trainer = Trainer(
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False),
    eval_metric = eval_metric
)


# 训练
trainer.train()


# 推理
pipe = pipeline("text-generation",model = model, tokenizer=tokenizer)
pipe("西安交通大学博物馆（Xi'an Jiaotong University Museum）是一座位于西安", max_length=128, do_sample=True)