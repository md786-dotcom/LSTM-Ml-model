# Model Files Information

The following model files are required but not included in this repository due to their size:

- `stock_price_lstm_model.h5` (~5-10 MB)
- `scaler_features.pkl` (~1-2 MB)
- `scaler_target.pkl` (~1 KB)

## How to Obtain Model Files

1. Run the Jupyter notebook `stock_price_prediction.ipynb` on Kaggle
2. After training completes, download the files from the Kaggle notebook output
3. Place the files in the root directory of this project

## Alternative: Git LFS

If you want to include model files in your repository:

1. Install Git LFS: https://git-lfs.github.com/
2. Run:
   ```bash
   git lfs install
   git lfs track "*.h5"
   git lfs track "*.pkl"
   git add .gitattributes
   git add *.h5 *.pkl
   git commit -m "Add model files with LFS"
   ```