# now that we have the cloud of points from the VAE 
# we can use them to randomly generate digits, like sampling random point from N(0,1 )
# then decode to image, so then we get VAE trained on N(0,1) distribution 

# there lot of things we can do -
# interpolation 
# we pass two digits, 3 and 8, we get the point cloud of something thats a blend of 3 and 8 so its a smooth morphing from 3 to 8
# WHY DOES THIS WORK?
# because latent space is CONTINOUS, path from z_a to z_b passes through valid regions, every point along the path decodes to somethign reasonable 
# like walking from cat to dog in concept space 

# Latent space arithmetic:
# we can literally treat latent space as our own version of math 
# we can treat LS as word embeddings, like if we feed latent(x) - latent(Y) to get latent(z) we can also do latent(k) + latent(z)  to get our desired output
# like combining attributes, remove attributes, scaling attributes etc 
# for example:
# we feed a image of 1 and feed a tilted digit of 7, then we get the vector so in latent space we subtract a new image of 9 from the precomputed vector 
# so then we have a new tilted image of 9 

# Conditional Generation:
# if we want a certain digit from mnist dataset 
# we can also just sample randomly from digits 
# and on each sample we see wher ethe cluster of 7 is, we keep caulcuating the loss and getting close to it 
# then we finally generate what we want 

# exploring dimensions 
# dims encode properties like roundness, vertical vs horizontal etc 
# each dimension learns a concept so we can do unsupervised kind of shit here 


# 
