# a vlm turns an image into a short sequence of vectors in the LANGUAGE model's embedding space and
# splices them into the text sequence at placeholder positions. the llm then treats them like tokens.
# shape story: pixels (b,3,H,W) -> vision encoder (b,n_patches,d_vis) -> projector (b,n_patches,d_text)
# -> scatter into text embeddings at image-token positions -> decoder.
# the number of placeholder tokens in the prompt MUST equal the number of image vectors.
# deps: torch. reread paligemma2/projector.py (a single linear) and paligemma2/paligemma.py
# (image_token_id mask and merge) and paligemma2/processor.py (how placeholders are inserted).
# explain: why scale or normalize image features before merging with text embeddings?
# explain: what are the cost tradeoffs of more image tokens (resolution) for prefill and kv cache?

import torch


def merge_image_features(text_embeds: torch.Tensor, input_ids: torch.Tensor,
                         image_feats: torch.Tensor, image_token_id: int) -> torch.Tensor:
    # text_embeds (b, t, d), input_ids (b, t), image_feats (b, n, d).
    # replace embeddings at positions where input_ids == image_token_id with image_feats, in order.
    # raise ValueError if any row's placeholder count != n. return a new tensor; do not mutate the input.
    pass


def image_token_budget(image_size: int, patch: int, n_images: int, text_tokens: int, max_ctx: int) -> dict:
    # return {"per_image": ..., "total_prompt": ..., "fits": bool, "text_left": ...}.
    # per_image = (image_size // patch) ** 2 tokens.
    pass


if __name__ == '__main__':
    # assert merge leaves non-placeholder positions bit-identical and fills placeholders in order.
    # assert a wrong placeholder count raises.
    pass
