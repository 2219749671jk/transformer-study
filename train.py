import torch
import torch.nn as nn
from model import make_model, subsequent_mask

class Batch:
    """封装一个batch数据，生成各类mask，切分decoder输入和标签"""
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # encoder屏蔽padding位置
        if tgt is not None:
            self.tgt = tgt[:, :-1]      # decoder输入序列，去掉末尾终止符
            self.tgt_y = tgt[:, 1:]     # 真实标签，去掉开头start符号
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum() # 统计有效非pad token数量

    @staticmethod
    def make_std_mask(tgt, pad):
        """组合两种mask：padding掩码 + 屏蔽未来token的上三角掩码"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask

class LabelSmoothing(nn.Module):
    """标签平滑：软化one‑hot标签，防止模型训练过度自信"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum") # 使用KL散度做损失
        self.padding_idx = padding_idx                 # pad不参与loss计算
        self.confidence = 1.0 - smoothing             # 正确类别保留的概率
        self.smoothing = smoothing                     # 分给其他类别的平滑系数
        self.size = size                               # 词表总大小
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2)) # 给非正确类别分配平滑概率
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) # 填充正确标签概率
        true_dist[:, self.padding_idx] = 0 # pad位置概率置0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.numel() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class SimpleLossCompute:
    """损失计算工具：decoder输出经过generator映射词表，再计算loss"""
    def __init__(self, generator, criterion):
        self.generator = generator   # 输出头：隐向量转词表概率
        self.criterion = criterion   # 损失函数对象

    def __call__(self, x, y, norm):
        x = self.generator(x)
        # 展平batch和序列维度，按有效token数归一化loss
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return loss

def rate(step, model_size, factor, warmup):
    """Transformer原论文Noam学习率策略：先warmup上升，之后逐步衰减"""
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode="train"):
    """执行一轮epoch：前向传播、计算loss，训练模式下反向传播更新参数"""
    total_tokens = 0
    total_loss = 0
    for batch in data_iter:
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask) # transformer整体前向
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train":
            optimizer.zero_grad()    # 清空上一轮梯度
            loss.backward()          # 反向传播求梯度
            optimizer.step()         # 更新模型参数
            scheduler.step()         # 更新学习率
        total_loss += loss.item() * batch.ntokens.item()
        total_tokens += batch.ntokens.item()
    return total_loss / total_tokens

def data_gen(V, batch_size, nbatches):
    """生成Copy任务数据集：输入序列与目标序列完全一致，用于验证transformer能否复现序列"""
    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1          # 序列第一个token固定为start起始标记
        src = data.clone().detach()
        tgt = data.clone().detach()
        yield Batch(src, tgt, pad=0)

if __name__ == "__main__":
    vocab_size = 11
    d_model = 512
    model = make_model(src_vocab=vocab_size, tgt_vocab=vocab_size, N=2) # N为encoder/decoder层数
    print("Transformer model created.")

    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(step, d_model, factor=1.0, warmup=400))
    loss_compute = SimpleLossCompute(model.generator, criterion)

    print("\nStart Training\n")
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        loss = run_epoch(data_gen(V=vocab_size, batch_size=32, nbatches=20), model, loss_compute, optimizer, scheduler)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | LR: {current_lr:.8f}")

    print("\nTraining Finished!")
    torch.save(model.state_dict(), "transformer_copy_model.pt") # 只保存模型权重，不保存网络结构
    print("Model saved: transformer_copy_model.pt")
