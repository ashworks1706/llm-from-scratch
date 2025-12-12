this is the third part of my llm architecture research journey

The Flow of Data:

Input ($x$): The token comes in from the previous floor.

Normalization 1: Clean up the data (RMSNorm).

MLA (Attention):

Compress input to latent.

Store in Cache.

Extract Meaning/Position.

Look at history.

Result: "I understand the context now."

Residual Connection 1: Add the result back to $x$.

Normalization 2: Clean up the data again (RMSNorm).

MoE (FeedForward):

Router looks at data.

Selects 6 out of 64 experts.

Experts process the data.

Result: "I have analyzed the meaning."

Residual Connection 2: Add result back to $x$.

Output: Send to the next floor.
