from command import configs
from fairseq.models.trex import TrexModel
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

trex = TrexModel.from_pretrained(f'checkpoints/similarity',
                                 checkpoint_file='checkpoint_best.pt',
                                 data_name_or_path=f'data-bin/similarity')
trex = trex.cuda()
trex.eval()
trex_script = torch.jit.script(trex.model)
trex_script.save('trex.pt')
loaded = torch.jit.load('trex.pt')

samples0 = {field: [] for field in configs.fields}
samples1 = {field: [] for field in configs.fields}
labels = []

for field in configs.fields:
    with open(f'data-src/similarity/valid.{field}.input0', 'r') as f:
        for line in f:
            samples0[field].append(line.strip())
for field in configs.fields:
    with open(f'data-src/similarity/valid.{field}.input1', 'r') as f:
        for line in f:
            samples1[field].append(line.strip())
with open(f'data-src/similarity/valid.label', 'r') as f:
    for line in f:
        labels.append(float(line.strip()))

top = 100
similarities = []

for sample_idx in range(top):
    sample0 = {field: samples0[field][sample_idx] for field in configs.fields}
    sample1 = {field: samples1[field][sample_idx] for field in configs.fields}
    label = labels[sample_idx]

    sample0_tokens = trex.encode(sample0)
    sample1_tokens = trex.encode(sample1)

    sample0_emb = trex.process_token_dict(sample0_tokens)
    sample1_emb = trex.process_token_dict(sample1_tokens)

    # emb0 = trex.predict('similarity', sample0_tokens)
    # emb1 = trex.predict('similarity', sample1_tokens)

    emb0 = loaded(sample0_emb, features_only=True)[0]['features']
    emb1 = loaded(sample1_emb, features_only=True)[0]['features']

    emb0_mean = torch.mean(emb0, dim=1)
    emb1_mean = torch.mean(emb1, dim=1)

    print(torch.cosine_similarity(emb0_mean, emb1_mean)[0].item())

    # directly predict pair
    concat_in = torch.cat((emb0_mean, emb1_mean, torch.abs(emb0_mean - emb1_mean), emb0_mean * emb1_mean), dim=-1)
    logits = loaded.classification_heads.similarity_pair(concat_in)

    # cosine similarity of function embedding
    emb0_rep = loaded.classification_heads.similarity(emb0)
    emb1_rep = loaded.classification_heads.similarity(emb1)
    similarities.append([torch.cosine_similarity(emb0_rep, emb1_rep)[0].item(), logits.argmax(dim=1).item(), label])

pred = np.array(similarities)
print(pred)
