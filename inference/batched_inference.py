# we compute attention via sequences
# now if we have multiple requests coming in that have different lengths, so to manage that we 
# add padding or empty token to batch so that it matches the shape
# so obviosuly the padding tokens are not supposed to be computed by attentions so thats why we had casual masking
# mask = [[1, 1, 0, 0, 0],  # first 2 are real
#            [1, 1, 1, 1, 1],  # all 5 are real
#            [1, 1, 1, 0, 0]]  # first 3 are real
