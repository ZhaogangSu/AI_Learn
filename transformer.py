# %%
import torch
import torch.nn as nn

# Positional Encoding
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_seq_len=5000) -> None:
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -torch.log(torch.tensor(10000.0))
        )
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :] # type: ignore
        

# Multi-head Attention
class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        
        self.head_dim = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = Q @ K.transpose(-1, -2)
        attn_weight = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        output = (attn_weight @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output
    
# Feed Forward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(nn.functional.gelu(self.fc1(x)))
    
# Transformer Block (LayerNorm + MHA + Res + LayerNorm + Fnn + Res)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.multi_head_attn = MultiHeadAttn(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.fnn = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.multi_head_attn(self.norm1(x)))
        x = x + self.dropout(self.fnn(self.norm2(x)))
        return x

# Transformer Encoder (repeated Transformer Block)
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_seq_len=5000) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPE(d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x

# %%
batch_size = 4
seq_len = 10

vocab_size = 10000
d_model = 512
n_heads = 8
d_ff = 2048
num_layers = 6
max_seq_len = 5000

head_dim = d_model // n_heads

model = TransformerEncoder(vocab_size, d_model, n_heads, d_ff, num_layers, max_seq_len)

token_ids = torch.randint(0, vocab_size, (2, 10, 512))
print(token_ids.shape)
# output = model(token_ids)

# print(output.shape)


