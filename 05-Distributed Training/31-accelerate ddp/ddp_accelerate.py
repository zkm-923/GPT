import torch
import pandas as pd
from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
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
    
    dataset = MyDataset()
    # 需要指定generator，保证每个进程的数据划分一致
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
    
    # 这里不需要指定sampler
    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)
    
    return trainloader, validloader


def prepare_model_and_optimizer():
    
    # 这里不需要将model指定到GPU的Global_Rank上
    model = BertForSequenceClassification.from_pretrained("/gemini/code/model")
    optimizer = Adam(model.parameters(), lr=2e-5)
    
    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            # 注意这里：汇总各个进程的pred和refs到一个进程来计算，防止出现样本数不是batch_szie的整数倍的问题
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, epoch=3, log_step=10):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            # loss.backward() ---> accelerator.backward(loss)
            accelerator.backward(loss)
            optimizer.step()
            if global_step % log_step == 0:
                # 汇总loss，dist.all_reduce(loss, op=dist.ReduceOp.AVG) --> loss = accelerator.reduce(loss, "mean")
                loss = accelerator.reduce(loss, "mean")
                # 打印：print_rank_0 --> accelerator.print
                accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate(model, validloader, accelerator)
        # print_rank_0 --> accelerator.print
        accelerator.print(f"ep: {ep}, acc: {acc}")


def main():

    # 初始化Accelerator
    accelerator = Accelerator()

    # 准备数据集
    trainloader, validloader = prepare_dataloader()

    # 准备model
    model, optimizer = prepare_model_and_optimizer()

    # 增加accelerator.prepare
    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)

    # 训练
    train(model, optimizer, trainloader, validloader, accelerator)


if __name__ == "__main__":
    main()
    # 启动命令：
        # torchrun --nproc_per_node=2 ddp_accelerate.py
        # accelerate launch ddp_accelerate.py
        # accelerate config，命令行进行参数选择，生成default_config,yaml文件，再输入accelerate launch ddp_accelerate.py
        # accelerate launch --help查看可以配置哪些信息