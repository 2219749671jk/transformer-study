import torch

from model import make_model, subsequent_mask


model = make_model(
    src_vocab=1000,
    tgt_vocab=1000,
    N=6
)

src = torch.randint(0, 1000, (2, 10))
tgt = torch.randint(0, 1000, (2, 8))

src_mask = torch.ones(2, 1, 10, dtype=torch.bool)
tgt_mask = subsequent_mask(8)

output = model(src, tgt, src_mask, tgt_mask)
log_probs = model.generator(output)

print("src shape:", src.shape)
print("tgt shape:", tgt.shape)
print("output shape:", output.shape)
print("generator output shape:", log_probs.shape)
print("probability sum:", torch.exp(log_probs[0, 0]).sum())