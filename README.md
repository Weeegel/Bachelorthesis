THESIS/
├── amazon/              # Experiments on the Amazon dataset
├── cora/                # Experiments on the Cora dataset
├── data/                # Raw and preprocessed datasets
├── minesweeper/         # Experiments on the Minesweeper dataset
├── PubMed/              # Experiments on the PubMed dataset
├── roman-empire/        # Experiments on the Roman Empire dataset 
├── src/                 
│   ├── methods/        
│   │   ├── ensemble_generator.py      # Generates degree-preserving rewiring & full-random edge rewiring
│   │   ├── posEnc.py                  # Laplacian positional encodings
│   │   └── randomPerturbations.py     # Random perturbations (edge perturbation, feature masking, feature noise)
│   └── models/          
│       ├── __init__.py
│       ├── gat.py        # Graph Attention Network
│       ├── gcn.py        # Graph Convolutional Network
│       ├── gin.py        # Graph Isomorphism Network
│       ├── graphCL_model.py  # Contrastive learning model
│       ├── graphSage.py  # GraphSAGE
│       ├── mlp.py        # Multi-Layer Perceptron & train method
│       └── svm.py        # SVM train method
├── plotting.py          # Plotting & visualization methods, method to save the results
├── trainMethoden.py     # Contrastive training loop & view generation




DATASETS:

    <dataset>/
    ├── results/                  # Saved results
    ├── baselines.ipynb           # Node2Vec & laplacian positional encodings & Original features as embeddings
    ├── dp_aug.ipynb              # Combination of degree-preserving rewiring & random perturbations
    ├── dp_rewire.ipynb           # Degree-preserving edge rewiring
    ├── plots.ipynb               # Visualizations and result plots
    ├── rand_aug.ipynb            # Random augmentations
    └── rand_rewire.ipynb         # Full random edge rewiring



