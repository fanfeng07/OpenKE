import torch
ent = torch.load('ent_embeddings.pt')
rel = torch.load('rel_embeddings.pt')
idx = torch.tensor([0])
print(ent)