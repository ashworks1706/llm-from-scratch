# normal autoencoders are just used for onstruction deconstructtion tasks like detectting, or images etc
# for autoencoders to also generate stuff, we need mor ethan just sampling, 
# thats what variational autoencoders do - 
# 'Generative' autoencoders
# the core differentiation is that instead of encoding to a POINT, we encode it to a distribution 
# such that
# in autoencoder we go -> image of '3' -> ENcoder -> z = [0.5,0.2] single point 
# in VAE, we go -> image of '3' -> encoder -> u = [0.5,0.2] mean, sigma = [0.1,0.1] std deviation 
# then we just sample form BOTH mean and std 
# so then we get cloud of points around mean '3'



# in VAE, we also get overlapping distributiosn that gives us conttinious smooth space 


# So the main architecture for autoencoder and VAE is :
# AE :
# x -> Encoder -> [mean] -> Decoder -> x 
# VAE :
# x -> Encoder -> [mean, std] -> sample z ~ N(mean, std^2) -> Decoder -> x 
# so we get single point in normal encoder but in encoder of VAE, we get mean and log variance (for numerical stability)


# notice how we used sampling from both mean and std, now since we're generating something, we need to backpropagate 
# and for that we need gradient, but sampling is random, so its not differentiable 
# the solution to this is to move randomness outside parameters, like normal sampling from normal number,
# and transform that deterministically by epsilon 
# so we can technically then call epsilon the randomness (independent of mean and std)
# so trasnform is deterministic thus, differentiable 
#
#
#
#
#
#
#
#



