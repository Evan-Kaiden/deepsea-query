import torch

class Embedder:
    def __init__(self, embedding_model, image_preprocess, tokenizer, device):
        self.embedding_model = embedding_model.to(device)
        self.image_preprocess = image_preprocess
        self.tokenizer = tokenizer
        self.device = device


        self.embedding_model.eval()

        for param in self.embedding_model.parameters():
            param.requires_grad_(False)

    def attn_bias(self, weights, lam=1.0):
        L = weights.numel() + 1
        bias = torch.zeros(L, L)
        bias[0, 1:] = lam * torch.log(weights.to(torch.float32))
        return bias.to(self.device)

    @torch.no_grad
    def embed_text(self, text):
        processed = self.tokenizer([text])
        processed = processed.to(self.device)
        return self.embedding_model.encode_text(processed).squeeze(0)

    @torch.no_grad
    def embed_image(self, image, attn_mask=None, lam=1.0, k=2):
        processed = self.image_preprocess(image)
        processed = processed.to(self.device)

        if attn_mask is None:
            return self.embedding_model.encode_image(processed.unsqueeze(0)).squeeze(0)

        v = self.embedding_model.visual
        bias = self.attn_bias(attn_mask, lam)

        x = v._embeds(processed.unsqueeze(0))
        blocks = v.transformer.resblocks
        for i, block in enumerate(blocks):
            x = block(x, attn_mask=bias if i >= len(blocks) - k else None)

        pooled, _ = v._pool(x)
        return (pooled @ v.proj).squeeze(0)


