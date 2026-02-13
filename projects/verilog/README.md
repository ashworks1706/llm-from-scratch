## The "Verified" Verilog Specialist

**Objective:** Fine-tune a model (Qwen-2.5-1.5B or 7B) to generate synthesizable Verilog by using a compiler as a "Judge."

### Tech Stack

* **Training:** Unsloth (QLoRA) + Hugging Face `trl`.
* **Data Validation:** `iverilog` (Icarus Verilog) + `yosys` (for logic synthesis checks).
* **Dataset:** Python scripts for scraping GitHub and OpenCores.
* **Deployment:** `llama.cpp` (GGUF) or `vLLM`.

### Implementation Steps

1. **Automated Curation:** Write a Python script to download `.v` files from GitHub.
2. **The "Linter" Filter:** Run a subprocess command: `iverilog -t null <file.v>`.
* If the exit code is non-zero, **discard it**. This ensures your model never "sees" a syntax error during training.
3. **Instruction Generation:** Use a "Teacher" model (like Llama-3-70B) to look at the clean code and describe it: *"Create a Verilog module for a 4-bit synchronous counter with asynchronous reset."*
4. **Fine-tuning:** Use Unsloth to train on these (Instruction, Verilog) pairs.
5. **Skills Gained:** Data Engineering, Programmatic Validation, QLoRA Hyperparameter Tuning.
