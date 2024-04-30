from neural_network import *
import torch as tr
from time import perf_counter
from test import test
db = tr.load("database.pt")
targs_train = tr.load("targs_train.pt")

from matplotlib import pyplot as pt
"""
Training loop
"""

num_updates = 1
length = len(db)
check_test = 500
batch_size = 64
loss_curve = []
accu_curve = []

stf = StackFormer(emb_dim)
# stf.use_flash_attention=True
model_name = "./model.pt"
stf.load_state_dict(tr.load(model_name))
loss_fn = tr.nn.CrossEntropyLoss()
opt = tr.optim.Adam(stf.parameters(), lr=0.0001)




print(length)

start = perf_counter()
def train():
  index = 0
  
  for update in range(num_updates):

    # prepare training batch
    stacks, goals, targs = [], [], []
    for b in range(batch_size):
      stacks_b = db[index]
      targs.append(targs_train[index])
      index += 1
      index = index % length
      goal_b = stacks_b[0] # inputs are stacks up to last step
      goals_b = [[goal_b]]

      stacks.append(stacks_b)
      goals += goals_b
      
      
    # forward
    goals, targs = tr.tensor(goals), tr.tensor(targs)
    logits = stf(stacks, goals)
    
    loss = loss_fn(logits, targs)
    loss_curve.append(loss.item())
    accu_curve.append((logits.argmax(dim=-1) == targs).to(float).mean().item())

    # backward
    opt.zero_grad()
    loss.backward()
    opt.step()

    # progress
    if update % 100 == 0: 
      print(f"update {update}: loss={loss_curve[-1]}, accu={accu_curve[-1]}")
      tr.cuda.empty_cache()
    if update % check_test == 0:
      test(loss_fn,stf)
    if update % 10000 == 0:
      tr.save(stf.state_dict(), model_name)
  

  print(f"total time = {perf_counter()-start}s")
  
train()

fig = pt.figure(figsize=(8,3))
pt.subplot(1,2,1)
pt.plot(loss_curve)
pt.ylabel("cross entropy loss")
pt.subplot(1,2,2)
pt.plot(accu_curve)
pt.ylabel("accuracy on batch")
fig.supxlabel("update")
_ = pt.tight_layout()
pt.show()