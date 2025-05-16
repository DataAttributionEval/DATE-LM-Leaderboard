# Sub-Tab Display
LEADERBOARD_NAMES = ["Pre-Training (10K)", 
                     "Pre-Training (30K)",
                     "Fine-Tuning", 
                     "Homogeneous", "Heterogeneous", 
                     "Factual Attribution"]

TRAINING_LEADERBOARDS = {"Pre-Training (10K)", "Pre-Training (30K)", "Fine-Tuning"}

# Submission Drop-Down Display
DROPDOWN_NAME_MAPPING = {"toxicity": {"Homogeneous", "Heterogeneous"},
                         "factual": {"Factual Attribution"},
                         "finetune": {"Fine-Tuning"},
                         "pretrain": {"Pre-Training (10K)", "Pre-Training (30K)"}}

# Leaderboard Columns
TOXICITY_COLS = ["Rank", "Method", "Attribution Method Type", "Model", "Model Size", "ToxicChat", "XSTest-response", "JailBreakBench", "AUPRC", "Paper/Code/Contact Link"]
FACTUAL_COLS = ["Rank", "Method", "Attribution Method Type", "Model", "Model Size", "Recall@50", "MRR", "Paper/Code/Contact Link"]
FINETUNE_COLS = ["Rank", "Method", "Attribution Method Type", "Model", "Model Size", "MMLU", "GSM8K", "BBH", "Paper/Code/Contact Link"]
PRETRAIN_COLS = ["Rank", "Method", "Attribution Method Type", "Model", "Model Size", "avg", "sciq", "arc_easy", "arc_challenge", "logiqa", "boolq", "hellaswag", "piqa", "winogrande", "openbookqa", "Paper/Code/Contact Link"]