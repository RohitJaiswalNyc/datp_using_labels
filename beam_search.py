import torch as tr
from metamathpy.proof import *
device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")
tr.set_default_device(device)
from helpers import *
from neural_network import StackFormer,emb_dim
from metamathpy.environment import Environment
from metamathpy.database import parse
import os


fpath = os.path.join("./set.mm")
db = parse(fpath)
embeddings = tr.load("./embeddings.pt")
env = Environment(db)
idx2label = list(embeddings)
test_labels = tr.load("./test_labels.pt")

model_name = "./model.pt"
stf = StackFormer(emb_dim)
stf.load_state_dict(tr.load(model_name))

def beam_search(stf, root_env, beam_size, max_depth):
  solution = None # successful proof if it has been found
  stf.eval()
  # initial environment for beam search
  beam = [root_env]
  log_probs = tr.zeros(len(beam))
  
  for depth in range(max_depth):

    # form current partial proofs into batch
    proofs = tr.tensor([[embeddings[tok] for tok in env.proof] for env in beam])
    goals = tr.tensor([[embeddings[env.claim.consequent.label]] for env in beam])
    
    # get stf predictions on entire beam as batch
    logits = stf(proofs, goals)
    # print(logits.shape," here")
    # logits = logits[:,-1,:] # auto-regression on last step of partial proofs

    # get log probabilities for proofs after each choice, broadcasts
    # print(logits.shape, log_probs.shape)
    log_probs = log_probs[:,None] + tr.nn.functional.log_softmax(logits, dim=-1)
    # print(log_probs.shape)
    
    # sort all predictions across beam from best to worst
    sort_idx = tr.argsort(log_probs.flatten(), descending=True)

    # convert flat index back to batch index and prediction
    # print(log_probs.shape)
    beam_idx, prediction = tr.unravel_index(sort_idx, log_probs.shape)

    # populate new beam from best to worst
    new_beam = []
    new_log_probs = []
    
    for (b, pred) in zip(beam_idx, prediction):
      pred = int(pred)
      rule_label = idx2label[pred]
      # print(f" b={b}, pred={rule_label} ({pred}), logprob={log_probs[b,pred]}")
      
      # stop if beam is full
      if len(new_beam) == beam_size: break

      # try applying rule
      env = beam[b].copy()
      # print(len(env.proof),len(env.stack))
      # print("stack=",[" ".join(x.conclusion) for x in env.stack])
      # print("labels=",[" ".join(x) for x in env.proof])
      # for x in env.stack:
      #   print(x)
      
      (_, proof, stack), msg = env.step(rule_label)

      
      # skip invalid rules and empty stacks
      if msg != "": continue
      if len(stack) == 0: continue

      # print(stack[-1].conclusion, " here " , tuple(env.claim.consequent.tokens))
      # if goal was predicted and matches stack, search is done
      
      if stack[-1].conclusion == tuple(env.claim.consequent.tokens): 
        solution = proof
        break

      # add to beam
      new_beam.append(env)
      new_log_probs.append(log_probs[b,pred])

    # overwrite previous iteration
    beam, log_probs = new_beam, tr.tensor(new_log_probs)

    # stop early if result has been proved
    if solution is not None: break

    # stop early if beam is empty (no valid rules predicted)
    if len(beam) == 0: break

  if solution == None:
    # print("solution not found")
    return 0
  else:
    # check result
    # print(f"checking solution found at depth {depth}:")
    # print(f"solution = {solution}")

    env = root_env.copy()
    for d, label in enumerate(solution):
      # print(f"{d}: {label}, stack:")
      # print(env.stack)

      # apply rule
      _, msg = env.step(label)
      if msg != "":
        # print(f"error: {msg}")
        print("hi")
        return 0

    # make sure proof succeeded
    assert stack[-1].conclusion == tuple(env.claim.consequent.tokens)
    return 1



# claim = db.rules["id"]
# proof_root, _ = verify_proof(db, claim)

# labels = extract_proof_labels(proof_root)



# env.reset(claim)
# env.step(labels[0])
# print(beam_search(stf, env, beam_size=10, max_depth=len(labels)))


accu = 0

for x in test_labels:
    # print(idx2label[x[-1][0]])
    claim = db.rules[x]
    
    proof_root = [x]
    root, _ = verify_proof(db, db.rules[x])
    labels = extract_proof_labels(root)
    
    env.reset(claim)
    env.step(labels[0])
    
    ret = beam_search(stf, env, beam_size=20, max_depth=len(labels))
    accu += ret
    print(f"lenth_of_proof= {len(labels)},proved={ret},proof={x}")
    
print(accu/10,accu)
