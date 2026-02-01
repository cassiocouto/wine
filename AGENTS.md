# Agent Onboarding Guide

This file helps onboard a new Claude agent to continue working on this project.

## Project Overview

**Wine Quality Prediction API** - A learning project to build a production-ready ML pipeline that predicts wine quality using scikit-learn models, exposed via FastAPI.

**User's Goal**: Learn ML fundamentals by building, not just consuming code. Act as a **tutor** - explain concepts, give hints, let them implement.

## Current State

### Completed
- [x] Project setup with `pyproject.toml` (editable install: `pip install -e ".[dev]"`)
- [x] Data exploration notebook (`notebooks/data_exploration.ipynb`)
  - Loaded wine quality dataset (red wine, semicolon delimiter)
  - Box plots for outlier detection
  - Distribution analysis (sulphates, residual sugar, chlorides are right-skewed)
  - Correlation heatmap
- [x] Data preparation module (`src/data/preparation.py`)
  - `normalize_dataframe()` - min-max scaling
  - `prepare_data_for_classification_as_np_arrays()` - with train/test split
- [x] Linear regression from scratch (`src/models/linear_regression_numpy.py`)
  - Implemented gradient descent (not Normal Equation)
  - User learned about convergence, learning rate, iterations
  - Achieved MSE ~0.386 (same as closed-form solution)
- [x] First PR created: `feature/eda-and-project-setup`

### In Progress / Next Steps
- [ ] **Model Training** (`src/models/train.py`) - Train multiple sklearn models (RF, GB, SVM, LogReg), compare with cross-validation, save best
- [ ] **Prediction Module** (`src/models/predict.py`) - Load saved models, singleton pattern
- [ ] **FastAPI Application** (`src/api/`)
  - Pydantic schemas for request/response
  - Endpoints: `/health`, `/predict/class`, `/predict/score`, `/predict`
- [ ] **Tests** (`tests/`)

## Key Files

```
wine/
├── data/wine_quality.csv         # Dataset (semicolon delimiter!)
├── notebooks/
│   ├── data_exploration.ipynb    # EDA notebook
│   └── linear_regression_numpy.ipynb  # Custom LR experiments
├── src/
│   ├── data/preparation.py       # Data loading, normalization, train/test split
│   ├── models/
│   │   └── linear_regression_numpy.py  # Custom LR implementation
│   └── utils/data_downloader.py  # Downloads dataset
├── plan.md                       # Learning guide (step-by-step instructions)
└── pyproject.toml                # Dependencies and project config
```

## Technical Notes

1. **Dataset**: Red wine quality from UCI/Kaggle. CSV uses semicolon (`;`) as delimiter.

2. **Feature scaling**: User implemented min-max normalization. For production, recommend `RobustScaler` due to outliers.

3. **Target variable**: `quality` (scores 3-8). For classification, bin into "low" (3-4), "medium" (5-6), "high" (7-9).

4. **Models planned**:
   - Classification: RandomForest, GradientBoosting, SVC, LogisticRegression
   - Regression: RandomForest, GradientBoosting, SVR, Ridge
   - Use cross-validation to select best

5. **No `gh` CLI installed** - GitHub operations need manual PR creation via web.

## User Preferences

- **Learning-focused**: Don't write code for them unless asked. Explain concepts, provide hints, let them implement.
- **Uses VS Code** with Jupyter notebooks
- **Restarts kernel** needed after modifying `src/` files (or use `%autoreload`)
- Language: Mix of English and Portuguese (saw "parei aqui" = "stopped here" in plan.md)
- **Update this file when requested**: When the user requests, update this file with new accomplishments, rules, and to-dos

## Useful Commands

```bash
# Install project
pip install -e ".[dev]"

# Run API (once implemented)
uvicorn src.api.main:app --reload

# Run tests
pytest tests/ -v

# Format code
black src/
```

## Recent Learnings Discussed

1. **Gradient Descent vs Normal Equation**: Both reach same optimal weights; GD is iterative, NE is closed-form
2. **Feature normalization**: Critical for gradient descent convergence
3. **Outliers**: Right-skewed features (residual sugar, chlorides) - tree models handle well, linear models may need RobustScaler
4. **pyproject.toml**: Modern Python packaging, replaces requirements.txt + setup.py
