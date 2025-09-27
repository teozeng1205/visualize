# ✈️ Airline Fare Family Clustering Analysis

A comprehensive data visualization and machine learning project analyzing airline fare family patterns across multiple major US carriers using advanced clustering and dimensionality reduction techniques.

## 🎯 Project Overview

This project performs sophisticated clustering analysis and visualization on airline fare family data from three major US carriers:
- **American Airlines (AA)**
- **JetBlue Airways (B6)** 
- **Alaska Airlines (AS)**

The analysis employs multiple dimensionality reduction techniques and clustering algorithms to uncover hidden patterns in fare family structures, pricing strategies, and route characteristics.

## 🚀 Features

### Advanced Dimensionality Reduction Methods
- **Principal Component Analysis (PCA)** - Linear dimensionality reduction
- **Kernel PCA** - Non-linear reduction with RBF and Polynomial kernels
- **UMAP** - Uniform Manifold Approximation and Projection for topology preservation
- **t-SNE** - t-Distributed Stochastic Neighbor Embedding for local structure
- **Factor Analysis** - Probabilistic latent factor modeling
- **Sparse PCA** - Feature selection with dimensionality reduction
- **Multidimensional Scaling (MDS)** - Distance preservation
- **Isomap** - Non-linear manifold learning
- **Locally Linear Embedding (LLE)** - Local neighborhood preservation
- **Linear Discriminant Analysis (LDA)** - Supervised dimensionality reduction

### Clustering Analysis
- **MiniBatch K-Means** clustering optimized for large datasets
- Cluster visualization across all dimensionality reduction spaces
- Automated cluster characteristic analysis

### Comprehensive Visualizations
- **Individual airline analysis** with 15+ visualization types per carrier
- **Combined multi-airline comparisons**
- **Fare family distribution analysis**
- **Price vs. duration relationship mapping**
- **Correlation heatmaps** for feature relationships
- **Box plots** for price distribution analysis

## 📊 Generated Visualizations

The project generates **70+ high-quality visualizations** including:

### Per-Airline Visualizations (AA, AS, B6)
- `pca_fare_families_{airline}.png` - PCA clustering by fare family
- `pca_clusters_{airline}.png` - PCA visualization of K-means clusters
- `umap_fare_families_{airline}.png` - UMAP manifold learning results
- `umap_clusters_{airline}.png` - UMAP cluster visualization
- `tsne_fare_families_{airline}.png` - t-SNE neighborhood preservation
- `tsne_clusters_{airline}.png` - t-SNE cluster mapping
- `kpca_rbf_fare_families_{airline}.png` - Kernel PCA with RBF kernel
- `kpca_poly_fare_families_{airline}.png` - Kernel PCA with polynomial kernel
- `spca_fare_families_{airline}.png` - Sparse PCA results
- `fa_fare_families_{airline}.png` - Factor Analysis visualization
- `mds_fare_families_{airline}.png` - Multidimensional Scaling
- `isomap_fare_families_{airline}.png` - Isomap manifold learning
- `lle_fare_families_{airline}.png` - Locally Linear Embedding
- `lda_fare_families_{airline}.png` - Linear Discriminant Analysis
- `fare_family_distribution_{airline}.png` - Distribution analysis
- `price_duration_{airline}.png` - Price vs. duration scatter plots
- `price_boxplot_{airline}.png` - Price distribution box plots
- `correlation_heatmap_{airline}.png` - Feature correlation analysis

### Cross-Airline Analysis
- `fare_family_clustering_analysis_ALL.png` - Comprehensive multi-airline comparison
- `dbscan_clusters_{airline}.png` - DBSCAN clustering results
- `anomaly_detection_{airline}.png` - Outlier detection visualization

## 🛠️ Technical Implementation

### Performance Optimizations
- **Aggressive sampling** strategy for large datasets (25K samples per airline)
- **Chunked data processing** with memory-efficient loading
- **MiniBatch algorithms** for scalable clustering
- **Parallel processing** capabilities for multiple visualizations
- **Smart sampling** for computationally intensive methods (t-SNE, MDS, Isomap)

### Data Processing Pipeline
1. **Multi-file CSV ingestion** with automatic brand detection
2. **Feature engineering** and categorical encoding
3. **Missing value imputation** using statistical methods
4. **Standardized scaling** for algorithm compatibility
5. **Dimensionality reduction** across 10+ methods
6. **Clustering analysis** with optimized parameters
7. **Visualization generation** with publication-quality output

## 📋 Requirements

```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
umap-learn>=0.5.0
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/teozeng1205/visualize.git
cd visualize

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Run the complete analysis pipeline
python clustering_visualization.py
```

The script will automatically:
1. Load and preprocess data from CSV files
2. Apply multiple dimensionality reduction techniques
3. Perform clustering analysis
4. Generate comprehensive visualizations
5. Output analysis summary

## 📁 Project Structure

```
visualize/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── clustering_visualization.py         # Main analysis script
├── .gitignore                         # Git ignore rules
│
├── Individual Airline Visualizations/
│   ├── *_AA.png                       # American Airlines plots
│   ├── *_AS.png                       # Alaska Airlines plots
│   └── *_B6.png                       # JetBlue Airways plots
│
└── Combined Analysis/
    └── fare_family_clustering_analysis_ALL.png
```

## 🔍 Key Insights

The analysis reveals:

- **Fare Family Clustering Patterns**: Distinct groupings across different carriers
- **Price-Duration Relationships**: Non-linear correlations varying by airline
- **Dimensionality Reduction Effectiveness**: UMAP and t-SNE show superior clustering separation
- **Cross-Airline Similarities**: Shared fare family structures across carriers
- **Outlier Detection**: Identification of unusual pricing or duration patterns

## 🎨 Visualization Quality

All visualizations feature:
- **High-resolution output** (200 DPI) for publication quality
- **Consistent color schemes** across related plots
- **Interactive legends** and detailed labeling
- **Grid overlays** for improved readability
- **Optimized transparency** for data density visualization

## 🔧 Customization

The script supports easy customization of:
- **Sample sizes** for performance tuning
- **Clustering parameters** (number of clusters, algorithms)
- **Visualization styles** and color schemes
- **Output formats** and resolution
- **Feature selection** for analysis

## 📈 Performance Metrics

- **Processing Time**: ~5-10 minutes for complete analysis
- **Memory Usage**: Optimized for systems with 8GB+ RAM
- **Output Size**: 70+ visualizations totaling ~50-100MB
- **Scalability**: Handles datasets up to 1M+ records per airline

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional dimensionality reduction techniques
- Interactive visualization capabilities
- Real-time data pipeline integration
- Advanced clustering algorithms
- Statistical significance testing

## 📄 License

This project is available for educational and research purposes. Please ensure compliance with data usage policies when working with airline fare data.

## 🙋‍♂️ Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

---

*This project demonstrates advanced machine learning techniques applied to airline industry data analysis, showcasing the power of dimensionality reduction and clustering for uncovering business insights.*
