speech codec covers how omni models hear and speak. audio is turned into discrete tokens with a neural codec so an llm can read and write it.

a residual vector quantizer compresses audio into a few codebooks per frame, and the first codebook is distilled to carry meaning. an audio encoder plus llm does speech recognition, and a thinker talker setup has the llm think in text while a small talker model produces speech tokens.
