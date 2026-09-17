import torch
from model import make_model, subsequent_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """贪心自回归解码：每次选择概率最大的token，拼到输入继续预测，直到达到最大长度"""
    memory = model.encode(src, src_mask)    # encoder对输入源序列编码，得到memory
    ys = torch.ones(1, 1, dtype=src.dtype, device=src.device).fill_(start_symbol) # decoder初始输入，start符号

    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).to(src.device) # decoder上三角mask，禁止偷看未来token
        out = model.decode(memory, src_mask, ys, tgt_mask)     # decoder前向计算
        prob = model.generator(out[:, -1])                     # 取出最后一个位置输出，预测下一个词
        _, next_word = torch.max(prob, dim=1)                  # 贪心，取概率最大的token索引
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)    # 将新token拼接到生成序列末尾
    return ys

if __name__ == "__main__":
    print("Transformer Inference")
    vocab_size = 11
    model = make_model(src_vocab=vocab_size, tgt_vocab=vocab_size, N=2) # 结构必须与训练完全一致
    model.load_state_dict(torch.load("transformer_copy_model.pt", map_location="cpu")) # 加载训练好权重
    model.eval()    # 推理模式，关闭dropout、bn等训练行为
    print("\nModel loaded successfully!")

    src = torch.tensor([[1, 3, 4, 5, 2]]) # 测试输入序列
    src_mask = (src != 0).unsqueeze(-2)   # encoder的padding mask

    with torch.no_grad():   # 推理阶段关闭梯度计算，节省显存加速计算
        result = greedy_decode(model, src, src_mask, max_len=src.size(1), start_symbol=1)

    print("\nInput sequence:")
    print(src)
    print("\nGenerated sequence:")
    print(result)
    print("\nExpected sequence:")
    print(src)

    if torch.equal(result, src):
        print("\nInference SUCCESS!")
    else:
        print("\nInference result is not perfect.")
