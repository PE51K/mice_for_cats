## Implementation Report

### Decisions that match the paper
- Dataset split sizes and per-API counts: demo 4,520; train 30/API (1,500); val 15/API (750); test 750 (STE).
- 8-shot ICL with greedy decoding; Llama models limited to `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct`.
- Confidence estimators and hyperparams: Raw Confidence; HRE (25 bins, width 0.04); NWKR (reflected Gaussian, auto bandwidth); MICE LR (L2=2 → C=0.5); MICE RF (n=1000, max_depth=20, max_features=10).
- Metrics: smECE, ETCU at τ={0.1,0.5,0.9}, AUC-ETCU. Seeds fixed to 42.

### Our implementations where the paper is underspecified
- Prompt: now loaded from `data/simulated-trial-and-error/STE/prompts/prompt_template.txt` (falling back to bundled template) and mirrors that structure; injects API descriptions/names and few-shot demos using Action / Action Input / Final Answer blocks.
- Demo selection: uses SentenceTransformer `sentence-transformers/paraphrase-mpnet-base-v2` (inferred from STE demo retrieval notebook), cosine similarity on L2-normalized embeddings, top-8 selection.
- Raw confidence token masking: exclude formatting tokens and truncate after the JSON arguments by heuristic brace matching.
- smECE/NWKR numerics: Silverman bandwidth, reflected Gaussian kernel, 1000-point grid for smECE; 999-point grid for ETCU AUC.
- Degenerate training labels: LR/RF/HRE/NWKR fall back to predicting the label mean if only one class is present.

### Additions beyond the paper
- Extra classification metrics (accuracy, precision/recall/F1, ROC-AUC, AP, Brier, log loss) reported alongside paper metrics.
- Cached feature IO and skip-generation flag for faster reruns.
