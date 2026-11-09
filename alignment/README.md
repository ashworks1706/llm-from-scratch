alignment covers understanding what models do inside and making them behave as intended.

interp, sae, steering, diffing and params are the mechanistic interpretability side. interp has the basic tools like patching, probes and lenses. sae finds features, steering changes behaviour with directions in the hidden states, diffing compares a model before and after finetuning, and params looks at the weights directly.

safety and oversight are the alignment side. safety covers how training goes wrong, like reward hacking and emergent misalignment, and how to fix it. oversight covers how to catch a model that misbehaves, like cot monitoring, ai control, sleeper agents and sandbagging.

explanability studies dataset and model behavior through data distribution, semantic patterns, linguistic features and token relationships.

a good order is interp first, then sae and steering, then safety and oversight, since the later lessons use probes, steering and the rl code from training. simple baselines like probes and prompting often beat fancy methods, so always compare against them.
