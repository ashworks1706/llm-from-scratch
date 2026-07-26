candle is the main model execution layer for this course. it is a rust native framework that can run real transformer models, load common weight formats and use cpu or cuda devices without requiring us to first rebuild every tensor and kernel primitive.

these lessons teach how to use candle deliberately rather than treating it as a black box. each lesson should inspect the tensor shapes, model artifacts, device placement, cache ownership and measured performance of the path being used.

the goal is a small but real rust llm runner that can load a model, generate tokens, support quantized execution and expose the state needed by the systems lessons.
