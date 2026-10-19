import torch
import torch.nn.functional as F


class Query:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store
    def query(self, num_return: int, image=None, text=None, attn_mask=None, **filters):

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

        embedding = F.normalize(embedding, dim=0)
        rows = self.store.search(embedding.cpu(), num_return, **filters)
        return [r[0] for r in rows], [r[2] for r in rows]






        

            
        