# convolutional operations from scratch
# the core operation that makes cnns work for images

# regular neural networks flatten out the images in the layers
# which results in way too many parameters, and it loses spatial structures ( treats nearby pixels same as for pixels)
# it also fails to detect centers or corners/ segmentation 


# in CNNs, instead of connectting every pixel to every neuron
# we use local connections and share weights (we share weights so that same filter across image)
# we also preserve spatial structure 
# this turns in fewer parameters and we do learn spatial awareness  for the same window we share 

















































