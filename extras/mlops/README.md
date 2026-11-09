mlops is the engineering around a model's whole life, not the model itself. the training folders here build models and inference serves them, but in real work you also need to track what you trained, version the data and weights, prove the model is actually good, wrap it as a service, and watch it once real traffic hits it. this folder covers that loop.

experiment tracking is the starting point. every training run has params, metrics and artifacts, and without logging them you can't tell which change helped or reproduce a good result later. covers mlflow and wandb and why a run is more than just a final number.

data and model versioning solves what git can't. git chokes on multi-gigabyte weights and datasets, so dvc stores the big files elsewhere and keeps lightweight pointers in git, giving you lineage from a result back to the exact data that made it.

model registry and packaging is how a checkpoint stops being a loose file. a registry gives models names, versions and stages (staging, production), and packaging bundles the weights with the code that runs them so the thing you evaluated is the exact thing you deploy. covers mlflow's registry and bentoml.

benchmarking and evals is how you earn the claim that a model is good. standardized suites like mmlu and hellaswag let you compare against the field on the same questions everyone else uses. covers lm-evaluation-harness and lighteval and how to read a benchmark number honestly.

llm eval and judges handles the tasks a fixed benchmark can't score. open-ended and rag outputs don't have one right answer, so we use task-specific metrics and llm-as-judge, and we build prompt regression tests so a change doesn't silently break yesterday's good behavior. covers ragas and deepeval and where these metrics lie.

observability and tracing is debugging in production. a single request can fan out into many model and tool calls, and when the answer is bad you need the trace, the tokens, the latency and the cost to see why. covers langfuse and arize phoenix.

serving as a service is turning a model into an api other things can call. this folder does the packaging and request-handling side with bentoml, and hands off to the inference folder for the actual high-performance engines like vllm.

distributed and scaling is what happens when one gpu isn't enough. covers spreading serving and batch jobs across workers and autoscaling with ray and ray serve, and when the added complexity is actually worth it.

pipelines and ci for ml wires the steps together. retrain, evaluate and deploy become a repeatable dag instead of manual steps, and ci gates a deploy on eval scores so a worse model can't ship. covers orchestration over mlflow and git.

monitoring and drift is the long tail after launch. real traffic drifts away from your training data, quality quietly degrades, and cost creeps, so this covers drift detection, guardrails, feedback loops and cost governance.

the point of this folder is that a model that only lives in a notebook isn't finished. mlops is what turns a checkpoint into something you can trust, ship, and improve.

these lessons use the python ml tooling installed in the shared learning venv, not the system python. the intended order is 01 through 10, first-principles: track, version, package, evaluate, observe, serve, scale, automate, monitor. each file holds its learning objectives, the imports you'll reach for, and a small entry point; it does not implement the lesson for you yet.
