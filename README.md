# Subset-kNN-Shapley

A Python library for computing Subset-kNN-Shapley classifiers. 
This library provides efficient algorithms for data valuation in machine learning, helping to understand the contribution of individual training samples to model predictions.

Main methods:
- `cod_subshap`: Inc-CkNN values
- `shapley_mc`: Monte Carlo Subset-kNN-Shapley values


## Installation

See `pyproject.toml` for dependencies.
We recommend using `uv` to install a local environment.

### Setup with uv

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and install all dependencies declared in `pyproject.toml`:

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

## Run tests with pytest

Run the test file with `pytest` from the project root:

```bash
pytest tests/test_all.py
```



## Quick Start

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Import methods
from algs.conditional_shapley import cod_subshap, build_faiss_index
from algs.mc import shapley_mc
from algs.baseline import knn_shapley_JW

# Generate sample data
X, y = make_classification(
    n_samples=200, 
    n_features=8, 
    n_classes=2, 
    n_informative=4,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Normalize features
X_mean, X_std = np.mean(X_train, 0), np.std(X_train, 0)
normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
X_train_norm, X_test_norm = normalizer_fn(X_train), normalizer_fn(X_test)

k = 5
F = [5]
b = 1
n_top = 100
ann_N = 150

print("Computing different value types...")

# 1. Inc-CkNN-E values
Inc_CkNN_E, _=cod_subshap(
    X_train_norm, y_train, X_test_norm, y_test, k, b, F
)
print(f"Inc-CkNN-E values: {Inc_CkNN_E[:5]}")

# 2. Inc-CkNN-Apx values  
Inc_CkNN_Apx, _=cod_subshap(
    X_train_norm, y_train, X_test_norm, y_test, k, b, F, n_top
)
print(f"Inc-CkNN-Apx values: {Inc_CkNN_Apx[:5]}")

# 3. Inc-CkNN-ANN values
ann_index = build_faiss_index(X_train_norm)
Inc_CkNN_ANN, _=cod_subshap(
    X_train_norm, y_train, X_test_norm, y_test, k, b, F, n_top, ann_index, ann_N
)
print(f"Inc-CkNN-ANN values: {Inc_CkNN_ANN[:5]}")

# 4. Monte Carlo subset-knn-shapley approximation
subset_knn_shapley_mc = shapley_mc(
    X_train_norm, y_train, X_test_norm, y_test, F, k, n_perms=1000
    )
print(f"MC subset-knn-shapley values: {subset_knn_shapley_mc[:5]}")   

# 5. knn-shapely values
knn_shapley = knn_shapley_JW(
     X_train_norm, y_train, X_test_norm, y_test, k
)
print(f"knn-shapley values: {knn_shapley[:5]}")   
```

Output:
```
Computing different value types...
Inc-CkNN-E values: [-0.01253532  0.07605505 -0.05771792 -0.096305    0.04437077]
Inc-CkNN-Apx values: [ 0.01044235  0.0608802  -0.0577962  -0.07295734  0.04342374]
Inc-CkNN-ANN values: [ 0.01044235  0.0608802  -0.0577962  -0.07295734  0.04342374]
MC subset-knn-shapley values: [-0.00091833  0.00403667 -0.0036925  -0.00468083  0.002315  ]
knn-shapley value:[-0.00069035  0.00491279 -0.00311579 -0.00486711  0.00199876]
```

## Dataset Information Documentation

This directory contains ten classic datasets for machine learning experiments. Below are detailed information for each dataset, including data types, sample counts, feature counts, and download links.

### 1. Adult Dataset (Census Income Prediction)

#### Basic Information
- **Task Type**: Binary classification (predicting annual income over $50K)
- **Data Format**: Tabular data (CSV format)
- **Sample Count**: 
  - Training set: 32,561 samples
  - Test set: 16,281 samples
  - Total: 48,842 samples
- **Feature Count**: 14 features (mixed types)

#### Feature Description
1. **age**: Age (continuous)
2. **workclass**: Work class (categorical)
3. **fnlwgt**: Final weight (continuous)
4. **education**: Education level (categorical)
5. **education-num**: Years of education (continuous)
6. **marital-status**: Marital status (categorical)
7. **occupation**: Occupation (categorical)
8. **relationship**: Family relationship (categorical)
9. **race**: Race (categorical)
10. **sex**: Gender (categorical)
11. **capital-gain**: Capital gains (continuous)
12. **capital-loss**: Capital losses (continuous)
13. **hours-per-week**: Hours worked per week (continuous)
14. **native-country**: Native country (categorical)

#### Target Variable
- **income**: Income category (<=50K or >50K)

#### Download Links
- **Original Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Direct Download**: [Adult Dataset](https://archive.ics.uci.edu/static/public/2/adult.zip)

### 2. CIFAR-10 Dataset

#### Basic Information
- **Task Type**: Image classification (10-class object recognition)
- **Data Format**: Image data (binary format)
- **Sample Count**: 
  - Training set: 50,000 images
  - Test set: 10,000 images
  - Total: 60,000 images
- **Image Size**: 32×32 pixel color images
- **Category Count**: 10 object categories

#### Category List
1. **airplane**: Airplane
2. **automobile**: Automobile
3. **bird**: Bird
4. **cat**: Cat
5. **deer**: Deer
6. **dog**: Dog
7. **frog**: Frog
8. **horse**: Horse
9. **ship**: Ship
10. **truck**: Truck

#### Data Characteristics
- 6,000 images per category
- Low image quality but suitable for rapid experiments
- Widely used for deep learning benchmarking

#### Download Links
- **Official Source**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Direct Download**: [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

### 3. MNIST Dataset (Handwritten Digit Recognition)

#### Basic Information
- **Task Type**: Multi-class classification (handwritten digit recognition)
- **Data Format**: Image data (binary IDX format)
- **Sample Count**: 
  - Training set: 60,000 images
  - Test set: 10,000 images
  - Total: 70,000 images
- **Image Size**: 28×28 pixel grayscale images
- **Category Count**: 10 classes (digits 0-9)

#### Category List
1. **0**: Zero
2. **1**: One
3. **2**: Two
4. **3**: Three
5. **4**: Four
6. **5**: Five
7. **6**: Six
8. **7**: Seven
9. **8**: Eight
10. **9**: Nine

#### Data Characteristics
- Balanced dataset with 6,000 images per digit in training set
- 1,000 images per digit in test set
- Grayscale handwritten digit images
- Widely used as the "hello world" of machine learning and computer vision

#### Download Links
- **Official Source**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

### 4. AG News Dataset (News Topic Classification)

#### Basic Information
- **Task Type**: Multi-class text classification (news topic classification)
- **Data Format**: Text data (CSV format)
- **Sample Count**: 
  - Training set: 120,000 news articles
  - Test set: 7,600 news articles
  - Total: 127,600 news articles
- **Category Count**: 4 news topics

#### Category Distribution
1. **World**: World news
2. **Sports**: Sports news
3. **Business**: Business news
4. **Sci/Tech**: Science and technology news

#### Feature Description
- **Class Index**: News category label (1-4)
- **Title**: News article title
- **Description**: News article content/description

#### Data Characteristics
- Balanced dataset with 30,000 samples per category in training set
- Each article contains title and description
- High-quality news text from AG's news corpus
- Widely used for text classification benchmarking

#### Download Links
- **Original Source**: [AG News Dataset](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- **Alternative Link**: [Kaggle AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

### 5. DBpedia Ontology Dataset

#### Basic Information
- **Task Type**: Multi-class text classification (ontology classification)
- **Data Format**: Text data (CSV format)
- **Sample Count**: 
  - Training set: 560,000 samples
  - Test set: 70,000 samples
  - Total: 630,000 samples
- **Category Count**: 14 ontology classes

#### Category Distribution
1. **Company**: Companies and corporations
2. **EducationalInstitution**: Educational institutions
3. **Artist**: Artists and performers
4. **Athlete**: Athletes and sportspeople
5. **OfficeHolder**: Government and political office holders
6. **MeanOfTransportation**: Vehicles and transportation
7. **Building**: Buildings and structures
8. **NaturalPlace**: Natural geographical places
9. **Village**: Villages and small settlements
10. **Animal**: Animals and species
11. **Plant**: Plants and vegetation
12. **Album**: Music albums
13. **Film**: Movies and films
14. **WrittenWork**: Books, articles, and written works

#### Feature Description
- **Class Index**: Category label (1-14)
- **Title**: Article title
- **Content**: Article content/description

#### Data Characteristics
- Balanced dataset with 40,000 samples per category in training set
- 5,000 samples per category in test set
- High-quality structured text data extracted from Wikipedia
- Widely used for text classification benchmarking

#### Download Links
- **Direct Download**: [dbpedia_csv.tar.gz](https://github.com/srhrshr/torchDatasets/raw/master/dbpedia_csv.tar.gz)
- **GitHub Source**: [DBpedia Dataset](https://emilhvitfeldt.github.io/textdata/reference/dataset_dbpedia.html?utm_source=chatgpt.com)

### 6. Amazon Review Polarity Dataset

#### Basic Information
- **Task Type**: Binary classification (sentiment analysis)
- **Data Format**: Text data (CSV format)
- **Sample Count**: 
  - Training set: 3,600,000 samples (1,800,000 positive, 1,800,000 negative)
  - Test set: 400,000 samples (200,000 positive, 200,000 negative)
  - Total: 4,000,000 samples
- **Category Count**: 2 classes (positive/negative)

#### Category Distribution
- **Class 1**: Negative reviews
- **Class 2**: Positive reviews

#### Feature Description
- **label**: Sentiment label (1 for negative, 2 for positive)
- **title**: Review title
- **text**: Review content
- **split**: Data split (train/test)

#### Data Characteristics
- Highly balanced dataset with equal number of positive and negative reviews
- Large-scale dataset with 4 million total samples
- Contains real Amazon customer reviews
- Widely used for sentiment analysis and text classification benchmarking

#### Download Links
- **Original Source**: [amazon review polarity](https://figshare.com/articles/dataset/Amazon_Review_Polarity/13232501?file=25484027)

### 7. IMDB Dataset (Movie Review Sentiment Analysis)

#### Basic Information
- **Task Type**: Binary classification (sentiment analysis)
- **Data Format**: Text data (CSV format)
- **Sample Count**: 50,000 movie reviews
- **Category Count**: 2 classes (positive/negative)

#### Category Distribution
- **positive**: Positive movie reviews
- **negative**: Negative movie reviews

#### Feature Description
- **review**: Movie review text
- **sentiment**: Sentiment label (positive/negative)

#### Data Characteristics
- Balanced dataset with 25,000 positive and 25,000 negative reviews
- Contains movie reviews from the Internet Movie Database (IMDB)
- Reviews are preprocessed and labeled for sentiment analysis
- Widely used for sentiment analysis and text classification benchmarking

#### Download Links
- **Original Source**: [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### 8. Covertype Dataset (Forest Cover Type)

#### Basic Information
- **Task Type**: Multi-class classification (predicting forest cover types)
- **Data Format**: Tabular data (compressed text format)
- **Sample Count**: 581,012 samples
- **Feature Count**: 54 features (12 original features, 54 encoded features)
- **Category Count**: 7 forest cover types

#### Feature Description
##### Quantitative Features (10)
1. **Elevation**: Elevation in meters
2. **Aspect**: Aspect in degrees azimuth
3. **Slope**: Slope in degrees
4. **Horizontal_Distance_To_Hydrology**: Horizontal distance to nearest surface water features
5. **Vertical_Distance_To_Hydrology**: Vertical distance to nearest surface water features
6. **Horizontal_Distance_To_Roadways**: Horizontal distance to nearest roadway
7. **Hillshade_9am**: Hillshade index at 9am, summer solstice
8. **Hillshade_Noon**: Hillshade index at noon, summer solstice
9. **Hillshade_3pm**: Hillshade index at 3pm, summer solstice
10. **Horizontal_Distance_To_Fire_Points**: Horizontal distance to nearest wildfire ignition points

##### Qualitative Features (44)
- **Wilderness_Area**: 4 wilderness areas (binary encoded)
- **Soil_Type**: 40 soil types (binary encoded)

#### Cover Types (Target Variable)
1. **Spruce/Fir**: Spruce/Fir
2. **Lodgepole Pine**: Lodgepole Pine
3. **Ponderosa Pine**: Ponderosa Pine
4. **Cottonwood/Willow**: Cottonwood/Willow
5. **Aspen**: Aspen
6. **Douglas-fir**: Douglas-fir
7. **Krummholz**: Krummholz

#### Download Links
- **Original Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/covertype)
- **Direct Download**: [covtype.data.gz](https://archive.ics.uci.edu/static/public/31/covtype.zip)

### 9. HIGGS Dataset (High Energy Physics)

#### Basic Information
- **Task Type**: Binary classification (signal vs background processes for Higgs boson detection)
- **Data Format**: Tabular data (compressed CSV format)
- **Sample Count**: Approximately 11 million samples
- **Feature Count**: 28 features (all numerical)
- **Category Count**: 2 classes (signal/background)

#### Feature Description
HIGGS dataset contains 28 features representing kinematic properties measured by particle detectors in high-energy physics experiments:

1. **Lepton pT**: Lepton transverse momentum
2. **Lepton eta**: Lepton pseudorapidity
3. **Lepton phi**: Lepton azimuthal angle
4. **Missing energy magnitude**: Missing transverse energy
5. **Missing energy phi**: Missing transverse energy azimuthal angle
6. **Jet 1 pT**: Leading jet transverse momentum
7. **Jet 1 eta**: Leading jet pseudorapidity
8. **Jet 1 phi**: Leading jet azimuthal angle
9. **Jet 1 b-tag**: Leading jet b-tagging discriminant
10. **Jet 2 pT**: Second jet transverse momentum
11. **Jet 2 eta**: Second jet pseudorapidity
12. **Jet 2 phi**: Second jet azimuthal angle
13. **Jet 2 b-tag**: Second jet b-tagging discriminant
14. **Jet 3 pT**: Third jet transverse momentum
15. **Jet 3 eta**: Third jet pseudorapidity
16. **Jet 3 phi**: Third jet azimuthal angle
17. **Jet 3 b-tag**: Third jet b-tagging discriminant
18. **Jet 4 pT**: Fourth jet transverse momentum
19. **Jet 4 eta**: Fourth jet pseudorapidity
20. **Jet 4 phi**: Fourth jet azimuthal angle
21. **Jet 4 b-tag**: Fourth jet b-tagging discriminant
22. **m_jj**: Invariant mass of two jets
23. **m_jjj**: Invariant mass of three jets
24. **m_lv**: Invariant mass of lepton and neutrino
25. **m_jlv**: Invariant mass of jet, lepton, and neutrino
26. **m_bb**: Invariant mass of two b-tagged jets
27. **m_wbb**: Invariant mass of W boson and two b-jets
28. **m_wwbb**: Invariant mass of WWbb system

#### Target Variable
- **Class**: Signal (1) vs Background (0) processes

#### Data Characteristics
- Large-scale dataset with ~11 million samples
- All features are numerical and continuous
- Simulated data from Monte Carlo simulations
- Highly imbalanced classes (signal events are rare)
- Used for benchmarking classification algorithms on large datasets

#### Download Links
- **Original Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS)


### 10. Skin Segmentation Dataset

#### Basic Information
- **Task Type**: Binary classification (skin vs non-skin detection)
- **Data Format**: Tabular data (LibSVM format)
- **Sample Count**: 245,057 samples
- **Feature Count**: 3 features (B, G, R color channels)
- **Category Count**: 2 classes (skin/non-skin)

#### Category Distribution
- **Skin**: 50,859 samples
- **Non-skin**: 194,198 samples

#### Feature Description
- **B**: Blue channel value
- **G**: Green channel value
- **R**: Red channel value

#### Data Characteristics
- Unbalanced dataset with skin samples comprising ~20.8% of total samples
- Color space: BGR (Blue, Green, Red)
- Values range from 0 to 255 for each channel
- Collected from face images of diverse age, gender, and race
- Used for skin detection in computer vision applications

#### Download Links
- **Original Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/229/skin+segmentation)

### Citation Information
If using these datasets for research, please cite the respective original sources.

---
*Dataset Information last updated: 2026-01-31*
