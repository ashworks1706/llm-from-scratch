# Engram hashed n gram memory (DeepSeek V4.1 Flash)
#
# LEARNING OBJECTIVES:
# - Hash the last few token ids into large embedding tables
# - Gate the looked up vectors into the hidden state
# - Keep the tables on cpu and show that it adds parameters without adding much compute
