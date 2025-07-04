# Sound Clustering Project

## Overview
This project implements unsupervised machine learning techniques to cluster unlabeled sound data using audio feature extraction and various clustering algorithms. The project analyzes 3,000 audio files (.wav format) to identify patterns and group similar sounds together.

## Project Structure
```
Sound Clustering/
├── Geu_Aguto_Sound_clustering_assignment.ipynb
├── Geu Aguto Garang_Hidden_Markov_Model_Capstone.pdf
├── README.md
└── unlabelled_sounds/ (3,010 .wav files)
```

## Features
- **Audio Feature Extraction**: Mel Spectrogram features using Librosa
- **Dimensionality Reduction**: PCA and t-SNE for visualization
- **Clustering Algorithms**: K-Means and DBSCAN
- **Performance Evaluation**: Silhouette Score and Davies-Bouldin Index
- **Data Visualization**: Comprehensive plots and analysis

## Technical Implementation

### Audio Feature Extraction
- **Mel Spectrogram Features**: 13-dimensional feature vectors
- **Parameters**:
  - n_mels: 13
  - n_fft: 2048
  - hop_length: 512
- **Processing**: Mean aggregation across time frames

### Machine Learning Pipeline
1. **Data Loading**: Efficient batch processing of 500 audio samples
2. **Feature Standardization**: StandardScaler normalization
3. **Dimensionality Reduction**: PCA and t-SNE for visualization
4. **Clustering**: K-Means and DBSCAN algorithms
5. **Evaluation**: Multiple clustering metrics

## Dependencies
```python
librosa>=0.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## Installation & Setup

### Google Colab (Recommended)
1. Open the notebook in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Geu-Pro2023/Sound_Clustering/blob/main/Geu_Aguto_Sound_clustering_assignment.ipynb)
2. Mount Google Drive to access audio files
3. Run all cells sequentially

### Local Environment
```bash
# Clone the repository
git clone https://github.com/Geu-Pro2023/Sound_Clustering.git
cd Sound_Clustering

# Install dependencies
pip install librosa numpy pandas matplotlib seaborn scikit-learn

# Launch Jupyter Notebook
jupyter notebook Geu_Aguto_Sound_clustering_assignment.ipynb
```

## Usage

### Basic Workflow
```python
# 1. Initialize feature extractor
extractor = AudioFeatureExtractor(n_mels=13)

# 2. Load and process audio data
features = extractor.load_dataset('path/to/audio/files', limit=500)

# 3. Apply clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features)

# 4. Evaluate results
silhouette_avg = silhouette_score(features, clusters)
```

### Key Components

#### AudioFeatureExtractor Class
- Extracts Mel Spectrogram features from audio files
- Handles batch processing with progress tracking
- Error handling for corrupted audio files

#### Clustering Analysis
- **K-Means**: Partitional clustering with configurable cluster count
- **DBSCAN**: Density-based clustering for noise detection
- **Evaluation Metrics**: Silhouette Score, Davies-Bouldin Index

## Results & Analysis

### Dataset Statistics
- **Total Audio Files**: 3,010 .wav files
- **Processed Samples**: 500 files (for computational efficiency)
- **Feature Dimensions**: 13 Mel Spectrogram coefficients
- **Data Shape**: (500, 13)

### Clustering Performance
The project evaluates clustering quality using:
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Lower values indicate better clustering
- **Visual Analysis**: t-SNE and PCA plots for cluster visualization

## Methodology

### 1. Data Preprocessing
- Audio loading with Librosa
- Mel Spectrogram computation
- Feature normalization and scaling

### 2. Dimensionality Reduction
- **PCA**: Linear dimensionality reduction for variance preservation
- **t-SNE**: Non-linear reduction for cluster visualization

### 3. Clustering Algorithms
- **K-Means**: Centroid-based clustering
- **DBSCAN**: Density-based clustering with noise detection

### 4. Evaluation & Visualization
- Quantitative metrics for cluster quality
- 2D visualizations of high-dimensional data
- Correlation analysis of audio features

## Key Insights
- High-dimensional audio features require dimensionality reduction for effective visualization
- Mel Spectrogram features capture essential audio characteristics for clustering
- Different clustering algorithms reveal various patterns in sound data
- Proper feature scaling is crucial for clustering performance

## Future Enhancements
- [ ] Implement additional audio features (MFCC, Chroma, Spectral features)
- [ ] Experiment with deep learning embeddings
- [ ] Add hierarchical clustering methods
- [ ] Implement real-time audio clustering
- [ ] Create interactive visualization dashboard

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments
- **Librosa**: Audio analysis library
- **Scikit-learn**: Machine learning algorithms
- **Google Colab**: Development environment
- **Audio Dataset**: Unlabeled sound collection for clustering analysis

## Author
**Geu Aguto**
- GitHub: [Geu-Pro2023](https://github.com/Geu-Pro2023)
- Project Repository: [Sound_Clustering](https://github.com/Geu-Pro2023/Sound_Clustering)

## Contact
For questions or collaboration opportunities, please reach out through GitHub or open an issue in the repository.

---
*This project demonstrates the application of unsupervised machine learning techniques to audio data analysis and clustering.*
