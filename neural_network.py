import torch as tr
import numpy as np
from matplotlib import pyplot as pt

device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")
tr.set_default_device(device)
embeddings = tr.load("./embeddings.pt")

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"



# constants
PAD_IDX = len(embeddings)
emb_dim = 512


def pad_trace(stacks):
  
  # print(stacks)
  # print(stacks, " here")
  max_len = max(map(len, stacks))
  padded = tr.full((len(stacks), max_len), PAD_IDX)
  pad_mask = tr.full(padded.shape, True)
  for s, stack in enumerate(stacks):
    padded[s,:len(stack)] = tr.tensor(stack)
    pad_mask[s,:len(stack)] = False
  return padded, pad_mask


def pe_tensor(d_model, max_len, base):
  pe = tr.zeros(max_len, d_model)
  position = tr.arange(0, max_len, dtype=tr.float).unsqueeze(1)
  div_term = tr.exp(tr.arange(0, d_model, 2).float() * (-np.log(base) / d_model))
  pe[:, 0::2] = tr.sin(position * div_term)
  pe[:, 1::2] = tr.cos(position * div_term)
  return pe

class PositionalEncoding(tr.nn.Module):

    def __init__(self, d_model, max_len=7224, base=10000.):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.register_buffer('pe', pe_tensor(d_model, max_len, base))

    def forward(self, x):
        seq_len = min(self.max_len, x.shape[1])
        x = x[:,-seq_len:] + self.pe[:seq_len, :]
        return x

class StackFormer(tr.nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.embedding = tr.nn.Embedding(num_embeddings=len(embeddings)+1, embedding_dim=d_model, padding_idx=PAD_IDX)
    self.pos_enc = PositionalEncoding(d_model)
    self.decoder = tr.nn.TransformerDecoderLayer(d_model, nhead=8, batch_first=True)

  def forward(self, stacks, goals):
    """
    stacks[b,s]: sth element of bth stack in the batch
    goals[b,0]: bth goal in the batch (singleton sequence)
    logits[b,r]: logit for rule r in bth example of the batch
    """
    # print(len(stacks),len(essentials)," before")
    
    # wrap input in tensors
    padded, pad_mask = pad_trace(stacks)


    # transformer
    memory = self.pos_enc(self.embedding(padded))
    queries = self.embedding(goals)
    # print(memory.shape,padded.shape,pad_mask.shape,queries.shape,memory.shape, "\n")
    # print(padded)
    result = self.decoder(queries, memory, memory_key_padding_mask=pad_mask) # (N, 1, E)

    # get logits for next rule
    readout = self.embedding.weight # (A+1, E)
    logits = (readout * result).sum(dim=-1) # (N, A+1)
    return logits
