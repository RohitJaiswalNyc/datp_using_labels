import torch as tr
import random
from neural_network import *
test_db = tr.load("./test.pt")
targs_test = tr.load("targs_test.pt")
from time import perf_counter
from matplotlib import pyplot as pt

num_updates = 374
batch_size = 64



def test(loss_fn,stf):
    index = 0
    stf.eval()
    loss_curve,accu_curve = [],[]
    for update in range(num_updates):

        # prepare training batch
        stacks, goals, targs = [], [], []
        for b in range(batch_size):
            stacks_b = test_db[index]
            targs.append(targs_test[index])
            index += 1
            index = index % len(test_db)
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
        print(f"update {update}: loss={loss_curve[-1]}, accu={accu_curve[-1]}")
    # progress
    
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

