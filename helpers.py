def extract_proof_labels(proof_root):
  labels = []
  rule = proof_root.rule
  if len(proof_root.dependencies) > 0:
    #for hyp in rule.essentials:
    for hyp in rule.floatings + rule.essentials:
      dep = proof_root.dependencies[hyp.label]
      sublabels = extract_proof_labels(dep)
      labels.extend(sublabels)

  labels.append(rule.consequent.label)
  return labels