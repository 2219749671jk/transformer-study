import torch

from model import make_model, subsequent_mask


# ============================================================
# Greedy Decode（贪心解码）
# ============================================================
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    使用训练好的 Transformer 进行贪心解码。

    核心思想：
    Decoder 每次只生成一个 token，
    并选择当前概率最大的 token 作为下一步输入。

    例如：
        已生成：[1]
        第一次预测：3

        已生成：[1, 3]
        第二次预测：4

        已生成：[1, 3, 4]
        第三次预测：5

        ...

    最终得到完整序列。
    """

    # --------------------------------------------------------
    # 1. Encoder 对输入序列进行编码
    #
    # src:
    # [batch_size, src_len]
    #
    # memory:
    # [batch_size, src_len, d_model]
    #
    # memory 会保存 Encoder 对输入序列提取出来的表示
    # --------------------------------------------------------
    memory = model.encode(
        src,
        src_mask
    )

    # --------------------------------------------------------
    # 2. 初始化 Decoder 输入
    #
    # Decoder 不能凭空开始生成，
    # 所以需要一个 START token。
    #
    # 本项目约定：
    # 0 = PAD
    # 1 = START
    # 2 = END
    # --------------------------------------------------------
    ys = torch.ones(
        1,
        1,
        dtype=src.dtype,
        device=src.device
    ).fill_(start_symbol)

    # --------------------------------------------------------
    # 3. 自回归生成
    #
    # Transformer Decoder 的特点：
    #
    # 生成第 2 个 token
    #       ↓
    # 使用前面的 token
    #       ↓
    # 再生成第 3 个 token
    #       ↓
    # 再使用前面的所有 token
    #
    # 一直重复
    # --------------------------------------------------------
    for i in range(max_len - 1):

        # ----------------------------------------------------
        # 创建 Decoder Mask
        #
        # 防止 Decoder 偷看未来 token。
        #
        # 这是 Transformer Decoder 非常重要的机制。
        # ----------------------------------------------------
        tgt_mask = subsequent_mask(
            ys.size(1)
        ).to(src.device)

        # ----------------------------------------------------
        # Decoder 前向传播
        #
        # memory：
        # Encoder 的输出
        #
        # ys：
        # 当前已经生成的 token
        # ----------------------------------------------------
        out = model.decode(
            memory,
            src_mask,
            ys,
            tgt_mask
        )

        # ----------------------------------------------------
        # 4. Generator
        #
        # out[:, -1]
        # 表示只取 Decoder 最后一个位置的输出。
        #
        # Generator 会把：
        #
        # d_model 维向量
        #
        # 转换成：
        #
        # vocab_size 个 token 的概率
        # ----------------------------------------------------
        prob = model.generator(
            out[:, -1]
        )

        # ----------------------------------------------------
        # 5. Greedy（贪心）
        #
        # 直接选择概率最大的 token。
        #
        # 例如：
        #
        # token 3 : 0.05
        # token 4 : 0.80   ← 最大
        # token 5 : 0.10
        #
        # 那么下一步就选择 token 4。
        # ----------------------------------------------------
        _, next_word = torch.max(
            prob,
            dim=1
        )

        # ----------------------------------------------------
        # 6. 把新预测出来的 token
        # 拼接到 Decoder 输入后面
        #
        # [1]
        # ↓
        # [1, 3]
        # ↓
        # [1, 3, 4]
        # ↓
        # [1, 3, 4, 5]
        # ----------------------------------------------------
        ys = torch.cat(
            [
                ys,
                next_word.unsqueeze(1)
            ],
            dim=1
        )

    return ys


# ============================================================
# inference.py 测试区
# ============================================================
if __name__ == "__main__":

    print("==============================")
    print("Transformer Inference")
    print("==============================")


    # ========================================================
    # 1. 模型参数
    #
    # !!! 非常重要 !!!
    #
    # 加载模型时的网络结构必须和训练时完全一致。
    #
    # train.py：
    # vocab_size = 11
    # N = 2
    #
    # inference.py 也必须一样。
    # ========================================================

    vocab_size = 11

    model = make_model(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        N=2
    )


    # ========================================================
    # 2. 加载训练好的模型参数
    #
    # train.py 中保存：
    #
    # torch.save(
    #     model.state_dict(),
    #     "transformer_copy_model.pt"
    # )
    #
    # 这里重新读取。
    # ========================================================

    model.load_state_dict(
        torch.load(
            "transformer_copy_model.pt",
            map_location="cpu"
        )
    )

    print("\nModel loaded successfully!")


    # ========================================================
    # 3. 切换到推理模式
    #
    # model.eval() 会关闭 Dropout 等训练阶段行为。
    #
    # 训练：
    # model.train()
    #
    # 推理：
    # model.eval()
    # ========================================================

    model.eval()


    # ========================================================
    # 4. 构造测试输入
    #
    # 我们现在做的是 Copy Task：
    #
    # 输入什么
    # ↓
    # Transformer 应该输出什么
    # ========================================================

    src = torch.tensor([
        [1, 3, 4, 5, 2]
    ])


    # ========================================================
    # 5. 创建 Encoder Mask
    #
    # 0 是 PAD token。
    #
    # 非 PAD：
    # True
    #
    # PAD：
    # False
    # ========================================================

    src_mask = (
        src != 0
    ).unsqueeze(-2)


    # ========================================================
    # 6. 开始推理
    #
    # torch.no_grad()：
    #
    # 推理阶段不需要：
    #     计算梯度
    #     backward
    #     更新参数
    #
    # 所以关闭梯度可以减少计算量和显存占用。
    # ========================================================

    with torch.no_grad():

        result = greedy_decode(
            model=model,
            src=src,
            src_mask=src_mask,
            max_len=src.size(1),
            start_symbol=1
        )


    # ========================================================
    # 7. 打印结果
    # ========================================================

    print("\nInput sequence:")
    print(src)

    print("\nGenerated sequence:")
    print(result)

    print("\nExpected sequence:")
    print(src)


    # ========================================================
    # 8. 判断 Copy Task 是否成功
    # ========================================================

    if torch.equal(result, src):

        print("\n==============================")
        print("Inference SUCCESS!")
        print("==============================")

    else:

        print("\n==============================")
        print("Inference result is not perfect.")
        print("==============================")