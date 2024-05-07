import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification

# 1、初始化进程组
import torch.distributed as dist
dist.init_process_group(backend="nccl")


# 2、导入数据
import pandas as pd
data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
data = data.dropna()


# 3、构建dataset
from torch.utils.data import Dataset
class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)
dataset = MyDataset()


# 4、数据切分，每个进程中的划分必须一致，否则训练集和测试集会有重复，会导致结果虚高。需要指定generator
from torch.utils.data import random_split
trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
for i in range(5):
    print(trainset[i])  # 通过打印数据可以验证在不同的进程中，划分的结果是否是一致的


# 5、分词
tokenizer = BertTokenizer.from_pretrained("/gemini/code/model")


# 6、构建data_loader，这里需要改动的是sampler，这样就可以完成分布式数据采样
def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# 注意这里使用了sampler，就没有指定shuffle=True，需要后面训练的时候自己实现
# 真实batch_size是32*2=64
trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(trainset))
validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, sampler=DistributedSampler(validset))
# next(enumerate(validloader))[1]


# 7、加载预训练模型
model = BertForSequenceClassification.from_pretrained("/gemini/code/model")


# 8、将模型放在当前GPU上，根据LOCAL_RANK环境变量来决定当前进程的model应该放在哪个GPU上
import os
from torch.nn.parallel import DistributedDataParallel as DDP
if torch.cuda.is_available():
    model = model.to(int(os.environ["LOCAL_RANK"]))
    
    
# 9、初始化DDP model
model = DDP(model)


# 10、模型训练
## 定义优化器
optimizer = Adam(model.parameters(), lr=2e-5)
# 在Global_RANK为0的GPU上打印
def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:
        print(info)
# 验证集评估
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    # 准确率需要汇总，这里使用dist.all_reduce计算每个进程的准确率，默认是计算和
    dist.all_reduce(acc_num)
    return acc_num / len(validset)
# 训练
def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        # 每一轮进行shuffle打乱顺序
        trainloader.sampler.set_epoch(ep)
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                # 对多个进程上的loss使用all_reduce计算均值
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                # 在Global_RANK为0的GPU上打印，不指定的话，会每个进程的结果都打印出来
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate()
        # 在Global_RANK为0的GPU上打印验证集的评估函数
        print_rank_0(f"ep: {ep}, acc: {acc}")
train()


# 11、执行代码
# 使用torchrun，并指定每个节点上的进程数
# torchrun --nproc_per_node=2 ddp.py