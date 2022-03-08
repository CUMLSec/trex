from command import configs
from fairseq.models.trex import TrexModel
import torch
import numpy as np
import csv

trex = TrexModel.from_pretrained(f'checkpoints/similarity',
                                 checkpoint_file='checkpoint_best.pt',
                                 data_name_or_path=f'data-bin/similarity')
trex = trex.cpu()
trex.eval()
trex_script = torch.jit.script(trex.model)
trex_script.save('checkpoints/similarity/trex.ptc')
loaded = torch.jit.load('checkpoints/similarity/trex.ptc')
# loaded = trex.model

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

top = 20
similarities = []

tp_cosine = fp_cosine = fn_cosine = tn_cosine = 0
tp_pair = fp_pair = fn_pair = tn_pair = 0

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

    emb0_rep = loaded(sample0_emb, features_only=True, classification_head_name='similarity')[0]['features']
    emb1_rep = loaded(sample1_emb, features_only=True, classification_head_name='similarity')[0]['features']
    # cosine similarity of function embedding
    # emb0_rep = loaded.classification_heads.similarity(emb0)
    # emb1_rep = loaded.classification_heads.similarity(emb1)

    # directly predict pair
    # emb0_mean = torch.mean(emb0, dim=1)
    # emb1_mean = torch.mean(emb1, dim=1)
    # concat_in = torch.cat((emb0_mean, emb1_mean, torch.abs(emb0_mean - emb1_mean), emb0_mean * emb1_mean), dim=-1)
    # logits = loaded.classification_heads.similarity_pair(concat_in)

    # print(emb0_rep[0, :2], emb1_rep[0, :2])
    pred_cosine = torch.cosine_similarity(emb0_rep, emb1_rep)[0].item()
    # pred_pair = logits.argmax(dim=1).item()

    # similarities.append([pred_cosine, pred_pair, label])
    similarities.append([pred_cosine, label])

with open('result/similarity.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(similarities)
