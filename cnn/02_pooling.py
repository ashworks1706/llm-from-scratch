# so in normal convolution what happens is that feature maps get larger and larger with conv layers increasing, this results in 
# too much computatio nand too many parameters, if we flatten for FC layer, its in millions, which also lets model to overfit eventually wihtout spatial awarenss
# pooling solves this by downsampling while keeping important informaiton
# there's two types of pooling called max pooling, average pooling and global average pooling 


# max pooling takes max value in each local region, why max? becasuse if a feature is detected, keep it
# exact position doesn't matter, just that it exists, strong activates = important features 
# 
