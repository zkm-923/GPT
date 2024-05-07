import os
import torch
import pandas as pd
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForSequenceClassification


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


def prepare_dataloader():

    # 构建dataset
    dataset = MyDataset()

    # 数据切分，每个进程中的划分必须一致，否则训练集和测试集会有重复，会导致结果虚高。需要指定generator随机种子
    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))

    tokenizer = BertTokenizer.from_pretrained("/gemini/code/model")

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    # 构建data_loader，这里需要改动的是sampler，这样就可以完成分布式数据采样
    # 注意这里使用了sampler，就没有指定shuffle=True，需要后面训练的时候自己实现
    # 真实batch_size是32*2=64
    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(trainset))
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, sampler=DistributedSampler(validset))

    return trainloader, validloader


def prepare_model_and_optimizer():
    
    model = BertForSequenceClassification.from_pretrained("/gemini/code/model")
    # 将模型放在当前GPU上，根据LOCAL_RANK环境变量来决定当前进程的model应该放在哪个GPU上
    if torch.cuda.is_available():
        model = model.to(int(os.environ["LOCAL_RANK"]))
    # 初始化DDP model
    model = DDP(model)
    optimizer = Adam(model.parameters(), lr=2e-5)
    return model, optimizer

# 在Global_RANK为0的GPU上打印
def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:
        print(info)


def evaluate(model, validloader):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    # 预测结果需要汇总，这里使用dist.all_reduce计算每个进程的准确率，默认是计算和
    dist.all_reduce(acc_num)
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, epoch=3, log_step=100):
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
        acc = evaluate(model, validloader)
        # 在Global_RANK为0的GPU上打印，不指定的话，会每个进程的结果都打印出来
        print_rank_0(f"ep: {ep}, acc: {acc}")


def main():
    
    # 初始化进程组
    dist.init_process_group(backend="nccl")

    # 准备数据集
    trainloader, validloader = prepare_dataloader()

    # 准备model
    model, optimizer = prepare_model_and_optimizer()

    # 训练
    train(model, optimizer, trainloader, validloader)


if __name__ == "__main__":
    main()
    # 执行命令
        # 使用torchrun，并指定每个节点上的进程数
        # torchrun --nproc_per_node=2 ddp.py