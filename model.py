import torch
import math
import copy
import torch.nn as nn
from torch.nn.functional import log_softmax


class EncoderDecoder(nn.Module): # 定义Transformer整体模型类，继承PyTorch神经网络基类nn.Module
    """
    标准的编码器-解码器架构,Transformer的总外壳
    """
    def __init__(self, encoder, decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder, self).__init__()  # 调用父类nn.Module的初始化方法
        self.encoder=encoder        # 保存编码器实例，负责编码源输入
        self.decoder=decoder        # 保存解码器实例，负责根据编码结果生成目标序列
        self.src_embed=src_embed    # 源序列嵌入层：token转为embedding+位置编码（输入侧）
        self.tgt_embed=tgt_embed    # 目标序列嵌入层：token转为embedding+位置编码（输出侧）
        self.generator=generator    # 输出头：把decoder输出向量映射到词表，计算单词概率

    def forward(self,src,tgt,src_mask,tgt_mask): # 模型前向传播入口，训练时自动调用，接收4个输入
        """
        接收原序列与目标序列,进行编码与解码
        """
        return self.decode( # 调用decode函数，返回解码后的结果
            self.encode(src,src_mask), # 参数1：调用encode，把源序列编码得到memory上下文向量
            src_mask, # 参数2：源序列掩码，用于交叉注意力，屏蔽padding占位符
            tgt, # 参数3：目标序列，已经生成的部分token
            tgt_mask # 参数4：目标序列掩码，防止decoder看到未来位置的token
        )

    def encode(self,src,src_mask): # 编码函数，专门处理源输入
        return self.encoder(self.src_embed(src),src_mask) # src先经过源嵌入层（词嵌入+位置编码），再送入encoder编码器

    def decode(self,memory,src_mask,tgt,tgt_mask): # 解码函数，利用编码得到的memory生成输出
        return self.decoder( # 送入decoder解码器
            self.tgt_embed(tgt), # 参数1：目标序列先做嵌入（词嵌入+位置编码）
            memory, # 参数2：encoder输出的上下文记忆向量，给交叉注意力读取源句子信息
            src_mask, # 参数3：源掩码，交叉注意力使用
            tgt_mask # 参数4：目标掩码，解码器自注意力用，屏蔽未来token
        )


class Generator(nn.Module): # 输出生成器类，继承PyTorch网络基类nn.Module，用来把decoder输出向量转为词表概率
    """
    定义标准线性层 + log softmax
    """
    def __init__(self,d_model,vocab): # 构造函数；d_model：模型隐藏层维度；vocab：词汇表总大小
        super(Generator,self).__init__() # 调用父类nn.Module的初始化方法，注册网络模块
        self.proj=nn.Linear(d_model,vocab) # 线性投影层：把d_model维的向量映射到vocab词表维度
    def forward(self,x): # 前向传播，x是decoder输出的特征向量，shape [batch, seq_len, d_model]
        return log_softmax(self.proj(x),dim=-1) # 先过线性层proj，再在最后一维做log_softmax，得到每个单词的对数概率


def clones(module,N): # 工具函数：复制N个完全独立相同的网络模块
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) # 列表推导式，深度拷贝N份module，包装进ModuleList给PyTorch管理



class Encoder(nn.Module): # 编码器类，由N层EncoderLayer堆叠而成
    def __init__(self, layer, N): # 构造函数：layer是单层编码器，N是堆叠层数（原论文N=6）
        super(Encoder,self).__init__() # 调用父类nn.Module初始化
        self.layers=clones(layer,N) # 调用clones函数，复制N层EncoderLayer保存到layers列表
        self.norm=LayerNorm(layer.size) # 最后一层输出后，做层归一化，layer.size就是d_model（512）

    def forward(self,x,mask): # 编码器前向传播，x是输入特征，mask是源序列padding掩码
        for layer in self.layers: # 循环遍历每一层EncoderLayer
            x=layer(x,mask) # 把x和mask送入当前层，输出作为下一层的输入
        return self.norm(x) # 全部N层计算完毕，执行层归一化，返回memory上下文向量



class LayerNorm(nn.Module): # 手动实现层归一化LayerNorm，和PyTorch内置LayerNorm功能一致
    def __init__(self,features,eps=1e-6): # features：特征维度d_model；eps极小值，防止除以0
        super(LayerNorm,self).__init__() # 调用父类nn.Module初始化
        self.a_2=nn.Parameter(torch.ones(features)) # 缩放参数γ，初始值全1，可训练
        self.b_2=nn.Parameter(torch.zeros(features)) # 偏移参数β，初始值全0，可训练
        self.eps=eps # 保存极小常数，防止标准差为0时分母报错

    def forward(self,x): # 前向传播，x输入张量 shape [batch, seq_len, features]
        mean=x.mean(-1,keepdim=True) # 在最后一维计算均值，keepdim保持维度不变用于广播
        std=x.std(-1,keepdim=True) # 在最后一维计算标准差，keepdim保持维度不变
        return self.a_2 *(x-mean)/(std+self.eps)+self.b_2 # LayerNorm计算公式：归一化后缩放+偏移



class SublayerConnection(nn.Module): # 子层连接模块：实现残差连接 + 前置层归一化(Pre-LN) + dropout，Transformer每个子层都要用
    def __init__(self,size,dropout): # size：d_model特征维度；dropout：dropout丢弃概率
        super(SublayerConnection,self).__init__() # 调用父类nn.Module初始化
        self.norm= LayerNorm(size) # 创建层归一化实例
        self.dropout = nn.Dropout(dropout) # dropout层，随机置零部分神经元防止过拟合

    def forward(self,x,sublayer): # x：子层输入；sublayer：一个可调用的子层函数（多头注意力 / FFN）
        return x + self.dropout(sublayer(self.norm(x))) # Pre-LN残差公式：先归一化，再过子层，dropout，最后加上原始输入x（残差连接）


class EncoderLayer(nn.Module):
    """
    Encoder 由 self-attention 和 feed forward 两个子层组成
    """
    def __init__(self, size, self_attn, feed_forward, dropout): # size:d_model；self_attn:多头自注意力模块；feed_forward:前馈网络；dropout：丢弃率
        super(EncoderLayer, self).__init__() # 调用父类构造函数

        self.self_attn = self_attn # 保存多头自注意力模块
        self.feed_forward = feed_forward # 保存FFN前馈网络模块
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # 克隆2份SublayerConnection，分别给注意力、FFN使用
        self.size = size # 保存特征维度d_model

    def forward(self, x, mask): # x：输入向量；mask：源序列padding掩码
        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, mask) # 第一个子层：自注意力；lambda包装，传入sublayer
        )
        return self.sublayer[1](x, self.feed_forward) # 第二个子层：前馈网络FFN，返回该层最终输出

class Decoder(nn.Module): # 解码器类，由N层DecoderLayer堆叠而成
    """
    解码器由 self-attention, src-attention, feed forward 三个子层组成
    """
    def __init__(self, layer, N): # layer：单层解码器；N：堆叠层数
        super(Decoder, self).__init__() # 调用父类nn.Module初始化
        self.layers = clones(layer, N) # 克隆N份DecoderLayer，保存到layers列表
        self.norm = LayerNorm(layer.size) # 最后一层输出后做层归一化

    def forward(self, x, memory, src_mask, tgt_mask): # x：目标序列输入；memory：编码器输出；src_mask：源掩码；tgt_mask：目标掩码
        for layer in self.layers: # 遍历每一层DecoderLayer
            x = layer(x, memory, src_mask, tgt_mask) # 送入当前层，得到输出作为下一层输入
        return self.norm(x) # 全部N层计算完毕，执行层归一化，返回最终解码结果

class DecoderLayer(nn.Module): # 解码器单层类
    """
    解码器单层由 self-attention, src-attention, feed forward 三个子层组成
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout): # size:d_model；self_attn:自注意力；src_attn:交叉注意力；feed_forward:前馈网络；dropout:丢弃率
        super(DecoderLayer, self).__init__() # 调用父类nn.Module初始化
        self.size = size # 保存特征维度d_model
        self.self_attn = self_attn # 保存自注意力模块
        self.src_attn = src_attn # 保存交叉注意力模块
        self.feed_forward = feed_forward # 保存前馈网络模块
        self.sublayer = clones(SublayerConnection(size, dropout), 3) # 克隆3份SublayerConnection，分别给三个子层使用

    def forward(self, x, memory, src_mask, tgt_mask): # x：目标序列输入；memory：编码器输出；src_mask：源掩码；tgt_mask：目标掩码
        m = memory # 保存编码器输出memory，用于交叉注意力读取源信息
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # 第一个子层：自注意力，屏蔽未来token
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) # 第二个子层：交叉注意力，读取编码器输出memory
        return self.sublayer[2](x, self.feed_forward) # 第三个子层：前馈网络FFN，返回该层最终输出

    
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'  缩放点积注意力
    """
    d_k = query.size(-1) # 获取query最后一维的维度，也就是单个头的维度d_k
    # 1. 计算q和k的点积，除以√d_k做缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 2. 如果有mask，把mask=0的位置填充成极小值-1e9
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 3. 最后一维做softmax，得到注意力权重
    p_attn = scores.softmax(dim=-1)
    # 4. 对注意力权重执行dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 5. 注意力权重和value相乘，返回加权结果 + 注意力权重矩阵
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 断言：d_model必须能被头数h整除，保证平分维度
        self.d_k = d_model // h  # 每个头的维度 = 总维度 / 头数，例如512/8=64
        self.h = h  # 注意力头的数量 h=8
        self.linears = clones(nn.Linear(d_model, d_model), 4) # 克隆4个相同的全连接层
        self.attn = None # 保存注意力权重矩阵，用于可视化
        self.dropout = nn.Dropout(p=dropout) # dropout层

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # 在mask增加一个维度，适配多头的维度 [B,1,seq,seq]
        nbatches = query.size(0) # 获取batch大小

        # 1. 对QKV做线性投影 + 分头
        query, key, value = [
            lin(x)                # 线性层映射，维度不变 d_model -> d_model
            .view(nbatches, -1, self.h, self.d_k) # reshape：[B, seq_len, h, d_k]
            .transpose(1, 2)      # 交换维度1和2 → [B, h, seq_len, d_k]，把头放到前面，方便并行计算
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2. 调用缩放点积注意力函数，所有头并行计算
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3. 多头结果拼接 concat
        x = (
            x.transpose(1, 2)          # 把头维度放回：[B,seq_len,h,d_k]
            .contiguous()              # 保证内存连续，view前必须加
            .view(nbatches, -1, self.h * self.d_k) # 合并多头：[B, seq_len, h*d_k] = [B,seq_len,d_model]
        )

        # 4. 最后一层线性映射，输出
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    逐位置前馈网FFN,Transformer编码器/解码器里的第二个子层
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)   # 第一层线性层：升维
        self.w_2 = nn.Linear(d_ff, d_model)   # 第二层线性层：降维
        self.dropout = nn.Dropout(dropout)    # dropout层

    def forward(self, x):
        # 计算流程：线性升维 → relu激活 → dropout → 线性降维
        return self.w_2(
            self.dropout(
                self.w_1(x).relu()
            )
        )


def subsequent_mask(size): # 生成目标序列的掩码矩阵，防止解码器看到未来token
    attn_shape=(1,size,size) # 生成一个1*size*size的张量
    subsequent_mask=torch.triu(torch.ones(attn_shape),diagonal=1).type(torch.uint8) # 上三角矩阵，主对角线以上为1
    return subsequent_mask==0 # 返回下三角矩阵，主对角线及

class Embeddings(nn.Module): # 词嵌入类，把token索引映射为d_model维的向量
    def __init__(self,d_model,vocab): # d_model：嵌入维度；vocab：词表大小
        super(Embeddings,self).__init__() # 调用父类nn.Module初始化
        self.lut=nn.Embedding(vocab,d_model) # 创建嵌入层，输入是词表索引，输出是d_model维向量
        self.d_model=d_model # 保存嵌入维度

    def forward(self,x): # 前向传播，x是token索引张量 shape [batch, seq_len]
        return self.lut(x)*math.sqrt(self.d_model) # 查表得到嵌入向量，并乘以√d_model做缩放

class PositionalEncoding(nn.Module): # 位置编码类，给序列中每个位置添加唯一的向量表示
    def __init__(self,d_model,dropout,max_len=5000): # d_model：嵌入维度；dropout：丢弃率；max_len：最大序列长度
        super(PositionalEncoding,self).__init__() # 调用父类nn.Module初始化
        self.dropout=nn.Dropout(p=dropout) # dropout层

        # 1. 创建位置编码矩阵pe，shape [max_len, d_model]
        pe=torch.zeros(max_len,d_model) # 初始化为全零
        position=torch.arange(0,max_len).unsqueeze(1).float() # 生成位置索引 [0,1,...,max_len-1]，shape [max_len,1]
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)) # 计算分母项，sin/cos公式中的10000^(2i/d_model)

        # 2. 按照公式计算sin和cos
        pe[:,0::2]=torch.sin(position*div_term) # 偶数维度使用sin
        pe[:,1::2]=torch.cos(position*div_term) # 奇数维度使用cos

        pe=pe.unsqueeze(0) # 增加batch维度，shape [1,max_len,d_model]
        self.register_buffer('pe',pe) # 注册为buffer，不参与梯度更新，但会随模型保存/加载

    def forward(self,x): # 前向传播，x是嵌入后的输入张量 shape [batch, seq_len, d_model]
        x=x+self.pe[:,:x.size(1)] # 将位置编码加到输入嵌入上，注意只取前seq_len个位置编码
        return self.dropout(x) # 返回加了位置编码的输入，并经过dropout

def make_model(
        src_vocab, tgt_vocab, N=6,
        d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """
    Helper: 构建一个标准的Transformer模型
    :param src_vocab: 源语言词表大小
    :param tgt_vocab: 目标语言词表大小
    :param N: 编码器/解码器堆叠层数
    :param d_model: 模型隐藏层维度
    :param d_ff: 前馈网络隐藏层维度
    :param h: 注意力头数
    :param dropout: dropout丢弃率
    :return: Transformer模型实例
    """
    c = copy.deepcopy # 方便深拷贝模块

    #  创建多头注意力模块和前馈网络模块
    attn = MultiHeadedAttention(h, d_model) # 多头注意力模块
    ff = PositionwiseFeedForward(d_model, d_ff, dropout) # 前馈网络模块
    position = PositionalEncoding(d_model, dropout) # 位置编码模块

    model=EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), # 编码器：N层堆叠
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N), # 解码器：N层堆叠
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)), # 源嵌入+位置编码
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), # 目标嵌入+位置编码
        Generator(d_model, tgt_vocab) # 输出生成器
    )
    for p in model.parameters(): # 初始化模型参数
        if p.dim() > 1: # 对于权重矩阵，使用Xavier均匀分布初始化
            nn.init.xavier_uniform_(p)
    return model # 返回构建好的Transformer模型


