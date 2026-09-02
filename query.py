import torch
import torch.nn.functional as F

from data_types import vector


class Query:
    def __init__(self, embedder):
        self.embedder = embedder
    def query(self, num_return:int, data_refs:list[vector], image=None, text=None, attn_mask=None):
        if image is None and text is None:
            raise ValueError

        got_text = text is not None
        got_image = image is not None

        if image is not None:
            img_embedding = self.embedder.embed_image(image, attn_mask)
            img_embedding =  F.normalize(img_embedding, dim=0)

        if text is not None:
            text_embedding = self.embedder.embed_text(text)
            text_embedding = F.normalize(text_embedding, dim=0)

        if got_text and got_image:
            combined = torch.stack([text_embedding, img_embedding], dim=-1)
            embedding = torch.mean(combined, dim=-1)
        elif got_text:
            embedding = text_embedding
        elif got_image:
            embedding = img_embedding

        # make this batched at some point
        sim_scores = []
        ref_ids = []
        for vector in data_refs:
            id = vector.id
            ref_embedding = F.normalize(vector.vector, dim=0)

            sim = F.cosine_similarity(ref_embedding, embedding, dim=0)

            sim_scores.append(sim)
            ref_ids.append(id)

        # sort
        sim_scores = torch.stack(sim_scores, dim=0) # (N,)
        sim_scores, indices =  torch.sort(sim_scores, descending=True)

        indices = indices[:num_return].tolist()
        sim_scores = sim_scores[:num_return]
        return [ref_ids[idx] for idx in indices], sim_scores


            






        

            
        