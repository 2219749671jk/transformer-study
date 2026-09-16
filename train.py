import torch
import torch.nn as nn

from model import make_model, subsequent_mask


# ============================================================
# 1. Batch：处理训练数据以及 Mask
# ============================================================

class Batch:
    """
    对一个 batch 的 src 和 tgt 进行预处理。

    主要完成：
    1. 创建 Encoder 的 src_mask
    2. 将目标序列拆成 Decoder 输入和正确答案
    3. 创建 Decoder 的 tgt_mask
    4. 统计真正参与 loss 计算的 token 数量
    """

    def __init__(self, src, tgt=None, pad=0):

        # Encoder 输入
        self.src = src

        # 屏蔽 PAD
        # shape:
        # [batch, seq_len]
        #       ↓
        # [batch, 1, seq_len]
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:

            # Decoder 输入：
            # 去掉最后一个 token
            #
            # 原始：
            # [1, 3, 4, 5, 2]
            #
            # Decoder 输入：
            # [1, 3, 4, 5]
            self.tgt = tgt[:, :-1]

            # Decoder 正确答案：
            # 去掉第一个 token
            #
            # [3, 4, 5, 2]
            self.tgt_y = tgt[:, 1:]

            # 创建 Decoder Mask
            self.tgt_mask = self.make_std_mask(
                self.tgt,
                pad
            )

            # 统计非 PAD token 数量
            # loss 最后需要用它进行归一化
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        创建 Decoder 使用的 mask。

        Decoder Mask 同时完成两件事情：

        1. 屏蔽 PAD
        2. 屏蔽未来 token（防止 Decoder 偷看答案）
        """

        # PAD Mask
        tgt_mask = (tgt != pad).unsqueeze(-2)

        # 与 Subsequent Mask 合并
        tgt_mask = tgt_mask & subsequent_mask(
            tgt.size(-1)
        ).type_as(tgt_mask)

        return tgt_mask


# ============================================================
# 2. Label Smoothing
# ============================================================

class LabelSmoothing(nn.Module):
    """
    标签平滑。

    普通 one-hot:
        正确类别 = 1
        其他类别 = 0

    Label Smoothing:
        正确类别保留较高概率
        一小部分概率分给其他类别

    这样可以避免模型过度自信。
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing=0.0
    ):

        super(LabelSmoothing, self).__init__()

        # 使用 KL Divergence 计算预测分布和目标分布之间的差异
        self.criterion = nn.KLDivLoss(
            reduction="sum"
        )

        self.padding_idx = padding_idx

        # 正确类别的概率
        self.confidence = 1.0 - smoothing

        # 平滑系数
        self.smoothing = smoothing

        # vocabulary size
        self.size = size

        # 保存平滑后的真实概率分布
        self.true_dist = None

    def forward(self, x, target):

        # x 的最后一维必须等于词表大小
        assert x.size(1) == self.size

        # 复制模型输出的 shape
        true_dist = x.data.clone()

        # 先给非正确类别分配 smoothing 概率
        true_dist.fill_(
            self.smoothing / (self.size - 2)
        )

        # 正确类别填入 confidence
        true_dist.scatter_(
            1,
            target.data.unsqueeze(1),
            self.confidence
        )

        # PAD token 不参与预测
        true_dist[:, self.padding_idx] = 0

        # 找到 target 中 PAD 的位置
        mask = torch.nonzero(
            target.data == self.padding_idx
        )

        # PAD 对应整行概率全部清零
        if mask.numel() > 0:
            true_dist.index_fill_(
                0,
                mask.squeeze(),
                0.0
            )

        self.true_dist = true_dist

        # 计算 KL Divergence Loss
        return self.criterion(
            x,
            true_dist.clone().detach()
        )


# ============================================================
# 3. Loss 计算器
# ============================================================

class SimpleLossCompute:
    """
    将 Decoder 输出经过 Generator，
    得到词表上的概率分布，然后计算 Loss。
    """

    def __init__(
        self,
        generator,
        criterion
    ):

        self.generator = generator
        self.criterion = criterion

    def __call__(
        self,
        x,
        y,
        norm
    ):

        # Decoder 输出：
        #
        # [batch, seq_len, d_model]
        #
        # Generator：
        #
        # [batch, seq_len, vocab_size]
        x = self.generator(x)

        # 展平成二维：
        #
        # [batch * seq_len, vocab_size]
        #
        # target：
        #
        # [batch * seq_len]
        loss = self.criterion(
            x.contiguous().view(
                -1,
                x.size(-1)
            ),
            y.contiguous().view(-1)
        ) / norm

        return loss


# ============================================================
# 4. Transformer Noam Learning Rate
# ============================================================

def rate(
    step,
    model_size,
    factor,
    warmup
):
    """
    Transformer 原论文中的学习率策略。

    前期：
        学习率逐渐增加Warmup

    后期：
        学习率逐渐下降

    参数：
        step       当前训练 step
        model_size Transformer 的 d_model
        factor     学习率缩放系数
        warmup     warmup steps
    """

    if step == 0:
        step = 1

    return factor * (
        model_size ** (-0.5)
        * min(
            step ** (-0.5),
            step * warmup ** (-1.5)
        )
    )


# ============================================================
# 5. 运行一个 Epoch
# ============================================================

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train"
):
    """
    完整运行一轮训练。

    流程：

    输入数据
        ↓
    Transformer Forward
        ↓
    Generator
        ↓
    Loss
        ↓
    Backward
        ↓
    Optimizer
        ↓
    Scheduler
    """

    total_tokens = 0
    total_loss = 0

    for batch in data_iter:

        # --------------------------------------------------------
        # 1. Transformer 前向传播
        # --------------------------------------------------------

        out = model(
            batch.src,
            batch.tgt,
            batch.src_mask,
            batch.tgt_mask
        )

        # --------------------------------------------------------
        # 2. 计算 Loss
        # --------------------------------------------------------

        loss = loss_compute(
            out,
            batch.tgt_y,
            batch.ntokens
        )

        # --------------------------------------------------------
        # 3. 反向传播
        # --------------------------------------------------------

        if mode == "train":

            # 清空上一轮梯度
            optimizer.zero_grad()

            # 反向传播，计算梯度
            loss.backward()

            # 根据梯度更新 Transformer 参数
            optimizer.step()

            # 更新学习率
            scheduler.step()

        # --------------------------------------------------------
        # 4. 统计 Loss
        # --------------------------------------------------------

        total_loss += (
            loss.item()
            * batch.ntokens.item()
        )

        total_tokens += batch.ntokens.item()

    return total_loss / total_tokens


# ============================================================
# 6. Copy Task 数据生成器
# ============================================================

def data_gen(
    V,
    batch_size,
    nbatches
):
    """
    自动生成 Copy Task 训练数据。

    Copy Task:

        输入：
        1 4 7 8 3 6 ...

        输出：
        1 4 7 8 3 6 ...

    模型需要学习：
        输入什么，就输出什么。

    这个任务主要用于验证 Transformer
    的训练流程是否正确。
    """

    for _ in range(nbatches):

        # 随机生成 token
        data = torch.randint(
            1,
            V,
            size=(batch_size, 10)
        )

        # 第一个 token 固定为 1
        # 可以理解成序列开始标志
        data[:, 0] = 1

        # Copy Task：
        # src 和 tgt 完全相同
        src = data.clone().detach()
        tgt = data.clone().detach()

        yield Batch(
            src,
            tgt,
            pad=0
        )


# ============================================================
# 7. 正式训练入口
# ============================================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )
    print("Create Transformer Model")
    print(
        "==============================\n"
    )

    # --------------------------------------------------------
    # 1. 基本参数
    # --------------------------------------------------------

    # Copy Task 的词表大小
    vocab_size = 11

    # Transformer 模型维度
    d_model = 512

    # --------------------------------------------------------
    # 2. 创建 Transformer
    # --------------------------------------------------------

    model = make_model(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,

        # 为了让学习实验运行更快，
        # 这里使用 2 层 Encoder / Decoder
        N=2
    )

    print("Transformer model created.")

    # --------------------------------------------------------
    # 3. 创建 Label Smoothing
    # --------------------------------------------------------

    criterion = LabelSmoothing(
        size=vocab_size,

        # Copy Task 中使用 0 作为 PAD
        padding_idx=0,

        smoothing=0.1
    )

    # --------------------------------------------------------
    # 4. Adam Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),

        # 真正学习率由 scheduler 控制
        lr=1.0,

        # Transformer 原论文常用参数
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # --------------------------------------------------------
    # 5. Noam Learning Rate Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,

        lr_lambda=lambda step: rate(
            step,
            model_size=d_model,
            factor=1.0,

            # 学习实验使用较小 warmup，
            # 可以更快观察训练效果
            warmup=400
        )
    )

    # --------------------------------------------------------
    # 6. 创建 Loss 计算器
    # --------------------------------------------------------

    loss_compute = SimpleLossCompute(
        model.generator,
        criterion
    )

    # --------------------------------------------------------
    # 7. 开始训练
    # --------------------------------------------------------

    print(
        "\n=============================="
    )
    print("Start Training")
    print(
        "==============================\n"
    )

    # 训练 20 个 Epoch
    num_epochs = 20

    for epoch in range(num_epochs):

        model.train()

        # 每一个 epoch 都重新随机生成训练数据
        loss = run_epoch(
            data_gen(
                V=vocab_size,
                batch_size=32,
                nbatches=20
            ),
            model,
            loss_compute,
            optimizer,
            scheduler,
            mode="train"
        )

        # 当前学习率
        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Loss: {loss:.4f} | "
            f"LR: {current_lr:.8f}"
        )

    # --------------------------------------------------------
    # 8. 训练完成
    # --------------------------------------------------------

    print(
        "\n=============================="
    )
    print("Training Finished!")
    print(
        "=============================="
    )

    # --------------------------------------------------------
    # 9. 保存模型参数
    # --------------------------------------------------------

    model_path = "transformer_copy_model.pt"

    torch.save(
        model.state_dict(),
        model_path
    )

    print("\nModel saved:")
    print(model_path)