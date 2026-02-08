# rearrange for readable tensor reshaping
# replacing cryptic view and permute chains

# topics to cover:
# - basic rearrange syntax 'b h w c -> b (h w) c'
# - splitting dimensions 'b (h w) -> b h w'
# - merging dimensions 'b h w c -> b (h w c)'
# - transposing 'b h w c -> b c h w'
# - comparing to pytorch view, reshape, permute
# - dimension names for clarity
# - pattern matching for validation
# - common mistakes and error messages

# OBJECTIVE: rewrite confusing reshapes from attention.py using rearrange
# see how code becomes self documenting
