#!/usr/bin/env python3
"""
Ultra-Efficient Clustering Visualization for Airline Fare Family Data

This script performs clustering analysis and dimensionality reduction on outbound_fare_family data
from multiple airline brand CSV files, creating beautiful visualizations with PCA, UMAP, and t-SNE.
Optimized for large datasets with aggressive sampling and parallel processing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, SparsePCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for efficiency
plt.rcParams['figure.max_open_warning'] = 0

class FareFamilyClusteringVisualizer:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        self.brand_mapping = {'AA': 'American Airlines', 'B6': 'JetBlue', 'AS': 'Alaska Airlines'}
        
    def load_and_preprocess_data(self, file_paths, max_samples_per_file=25000):
        """Load and preprocess data from multiple CSV files with aggressive sampling for efficiency"""
        print("Loading and preprocessing data with aggressive sampling...")
        
        # Load data efficiently with specific dtypes and chunking
        base_dtype = {
            'cabin': 'category',
            'outbound_fare_family': 'category', 
            'price_inc': 'float32',
            'duration': 'int16',
            'refundable': 'bool',
            'avg_price_per_min': 'float32',
            'total_itineraries': 'int16'
        }
        optional_dtype = {
            'ap': 'int16',
            'outbound_gcm': 'int32',
            'change_fee': 'float32'
        }
        
        dataframes = []
        for file_path in file_paths:
            # Extract brand from filename
            brand = file_path.split('_')[1].split('.')[0]
            print(f"Processing {brand}...")
            
            # Read file in chunks and sample efficiently
            chunk_size = 50000
            chunks = []
            
            # Detect available columns to build safe dtype/usecols
            header_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
            dtype_per_file = {k: v for k, v in base_dtype.items() if k in header_cols}
            dtype_per_file.update({k: v for k, v in optional_dtype.items() if k in header_cols})
            usecols = list(dtype_per_file.keys())
            
            for chunk in pd.read_csv(file_path, dtype=dtype_per_file, usecols=usecols, chunksize=chunk_size):
                # Sample from each chunk to maintain representativeness
                sample_size = min(len(chunk), max(1000, len(chunk) // 10))
                chunk_sample = chunk.sample(n=sample_size, random_state=42)
                chunks.append(chunk_sample)
                
                # Stop if we have enough samples
                if sum(len(c) for c in chunks) >= max_samples_per_file:
                    break
            
            # Combine chunks and take final sample
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                if len(df) > max_samples_per_file:
                    df = df.sample(n=max_samples_per_file, random_state=42)
                
                df['brand'] = brand
                df['brand_name'] = self.brand_mapping.get(brand, brand)
                dataframes.append(df)
                print(f"Sampled {len(df):,} records from {brand}")
        
        # Combine all data
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"Total records for analysis: {len(self.data):,}")
        
        # Encode categorical variables efficiently
        le_cabin = LabelEncoder()
        le_fare_family = LabelEncoder()
        
        self.data['cabin_encoded'] = le_cabin.fit_transform(self.data['cabin'].astype(str))
        self.data['fare_family_encoded'] = le_fare_family.fit_transform(self.data['outbound_fare_family'].astype(str))
        
        # Create feature matrix for clustering
        feature_cols = ['cabin_encoded', 'fare_family_encoded', 'price_inc', 'duration', 
                       'refundable', 'avg_price_per_min', 'total_itineraries']
        # Include optional features if present
        for opt_col in ['ap', 'outbound_gcm', 'change_fee']:
            if opt_col in self.data.columns:
                feature_cols.append(opt_col)
        
        self.features = self.data[feature_cols].copy()
        
        # Handle missing values efficiently
        self.features = self.features.fillna(self.features.median())
        
        # Convert boolean to int for efficiency
        self.features['refundable'] = self.features['refundable'].astype(int)
        
        # Scale features
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(self.features)
        
        # Store label encoders for later use
        self.le_cabin = le_cabin
        self.le_fare_family = le_fare_family
        
        return self.data
    
    def perform_clustering(self, n_clusters=6):
        """Perform efficient clustering analysis"""
        print("Performing efficient clustering analysis...")
        
        # Use MiniBatchKMeans for efficiency with large datasets
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, max_iter=100)
        self.data['kmeans_cluster'] = kmeans.fit_predict(self.features_scaled)
        
        print(f"MiniBatch K-Means found {n_clusters} clusters")
        
        return self.data
    
    def apply_dimensionality_reduction(self):
        """Apply comprehensive dimensionality reduction methods for advanced visualization"""
        print("Applying comprehensive dimensionality reduction methods...")
        
        # 1. Standard PCA (Linear)
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(self.features_scaled)
        self.data['pca_1'] = pca_result[:, 0]
        self.data['pca_2'] = pca_result[:, 1]
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
        
        # 2. Kernel PCA (Non-linear) - RBF kernel
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, random_state=42)
        kpca_result = kpca.fit_transform(self.features_scaled)
        self.data['kpca_1'] = kpca_result[:, 0]
        self.data['kpca_2'] = kpca_result[:, 1]
        
        # 3. Kernel PCA - Polynomial kernel
        kpca_poly = KernelPCA(n_components=2, kernel='poly', degree=3, random_state=42)
        kpca_poly_result = kpca_poly.fit_transform(self.features_scaled)
        self.data['kpca_poly_1'] = kpca_poly_result[:, 0]
        self.data['kpca_poly_2'] = kpca_poly_result[:, 1]
        
        # 4. Sparse PCA
        spca = SparsePCA(n_components=2, random_state=42, alpha=0.1)
        spca_result = spca.fit_transform(self.features_scaled)
        self.data['spca_1'] = spca_result[:, 0]
        self.data['spca_2'] = spca_result[:, 1]
        
        # 5. Factor Analysis (Probabilistic approach)
        fa = FactorAnalysis(n_components=2, random_state=42)
        fa_result = fa.fit_transform(self.features_scaled)
        self.data['fa_1'] = fa_result[:, 0]
        self.data['fa_2'] = fa_result[:, 1]
        
        # 6. UMAP (efficient for visualization)
        umap_reducer = umap.UMAP(
            n_components=2, 
            random_state=42, 
            n_neighbors=10,
            min_dist=0.1,
            n_epochs=200,
            verbose=False
        )
        umap_result = umap_reducer.fit_transform(self.features_scaled)
        self.data['umap_1'] = umap_result[:, 0]
        self.data['umap_2'] = umap_result[:, 1]
        
        # 7. MDS (Multidimensional Scaling) - sample for efficiency
        sample_size_mds = min(8000, len(self.data))
        mds_data = self.data.sample(n=sample_size_mds, random_state=42)
        mds_features = self.features_scaled[mds_data.index]
        
        mds = MDS(n_components=2, random_state=42, max_iter=300, n_init=1)
        mds_result = mds.fit_transform(mds_features)
        
        mds_data_copy = mds_data.copy()
        mds_data_copy['mds_1'] = mds_result[:, 0]
        mds_data_copy['mds_2'] = mds_result[:, 1]
        self.mds_data = mds_data_copy
        
        # 8. Isomap (Non-linear manifold learning)
        sample_size_iso = min(5000, len(self.data))
        iso_data = self.data.sample(n=sample_size_iso, random_state=42)
        iso_features = self.features_scaled[iso_data.index]
        
        isomap = Isomap(n_components=2, n_neighbors=10)
        iso_result = isomap.fit_transform(iso_features)
        
        iso_data_copy = iso_data.copy()
        iso_data_copy['iso_1'] = iso_result[:, 0]
        iso_data_copy['iso_2'] = iso_result[:, 1]
        self.iso_data = iso_data_copy
        
        # 9. Locally Linear Embedding (LLE)
        sample_size_lle = min(5000, len(self.data))
        lle_data = self.data.sample(n=sample_size_lle, random_state=42)
        lle_features = self.features_scaled[lle_data.index]
        
        try:
            lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
            lle_result = lle.fit_transform(lle_features)
            
            lle_data_copy = lle_data.copy()
            lle_data_copy['lle_1'] = lle_result[:, 0]
            lle_data_copy['lle_2'] = lle_result[:, 1]
            self.lle_data = lle_data_copy
            self.has_lle = True
        except:
            print("LLE failed - skipping")
            self.has_lle = False
        
        # 10. t-SNE (aggressive sampling for efficiency)
        sample_size_tsne = min(5000, len(self.data))
        tsne_data = self.data.sample(n=sample_size_tsne, random_state=42)
        tsne_features = self.features_scaled[tsne_data.index]
        
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=min(30, sample_size_tsne//4),
            max_iter=300,
            verbose=0
        )
        tsne_result = tsne.fit_transform(tsne_features)
        
        tsne_data_copy = tsne_data.copy()
        tsne_data_copy['tsne_1'] = tsne_result[:, 0]
        tsne_data_copy['tsne_2'] = tsne_result[:, 1]
        self.tsne_data = tsne_data_copy
        
        # 11. Linear Discriminant Analysis (supervised)
        if self.data['outbound_fare_family'].nunique() > 2:
            try:
                lda = LinearDiscriminantAnalysis(n_components=2)
                lda_result = lda.fit_transform(self.features_scaled, self.data['fare_family_encoded'])
                self.data['lda_1'] = lda_result[:, 0]
                self.data['lda_2'] = lda_result[:, 1] if lda_result.shape[1] > 1 else np.zeros(len(lda_result))
                self.has_lda = True
                print(f"LDA explained variance ratio: {lda.explained_variance_ratio_.round(3) if hasattr(lda, 'explained_variance_ratio_') else 'N/A'}")
            except:
                print("LDA failed - skipping")
                self.has_lda = False
        else:
            self.has_lda = False
        
        print("Comprehensive dimensionality reduction completed")
        
    def create_visualizations(self):
        """Create beautiful and efficient visualizations for all airlines"""
        print("Creating efficient visualizations for all airlines...")
        
        # Use Agg backend for efficiency (no display required)
        plt.switch_backend('Agg')
        
        # Create individual plots for each airline
        self.create_individual_airline_plots()
        
        # Create combined overview plot
        self.create_combined_overview_plot()
        
    def create_individual_airline_plots(self):
        """Create individual plots for each airline"""
        print("Creating individual airline plots...")
        
        for brand in self.data['brand'].unique():
            brand_data = self.data[self.data['brand'] == brand]
            brand_name = self.brand_mapping[brand]
            
            print(f"Creating plots for {brand_name}...")
            
            # Create color mapping for fare families for this airline
            unique_families = sorted(brand_data['outbound_fare_family'].unique())
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_families)))
            family_colors = dict(zip(unique_families, colors))
            
            # Sample data for plotting efficiency
            plot_data = brand_data.sample(n=min(15000, len(brand_data)), random_state=42) if len(brand_data) > 15000 else brand_data
            
            # Get top fare families for this airline
            top_families = plot_data['outbound_fare_family'].value_counts().head(10).index
            
            # Create individual subplot PNGs
            self.create_individual_subplots(brand, brand_name, plot_data, family_colors, top_families)
            
    def create_individual_subplots(self, brand, brand_name, plot_data, family_colors, top_families):
        """Create individual subplot PNGs for each visualization type"""
        
        # 1. PCA by Fare Family
        plt.figure(figsize=(12, 8))
        for family in top_families:
            family_data = plot_data[plot_data['outbound_fare_family'] == family]
            if len(family_data) > 0:
                plt.scatter(family_data['pca_1'], family_data['pca_2'], 
                           c=[family_colors[family]], label=family, 
                           alpha=0.7, s=12, edgecolors='none')
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.title(f'PCA: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'pca_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 2. PCA by Cluster
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(plot_data['pca_1'], plot_data['pca_2'], 
                            c=plot_data['kmeans_cluster'], cmap='tab10', 
                            alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.title(f'PCA: K-Means Clustering Results ({brand_name})', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'pca_clusters_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 3. UMAP by Fare Family
        plt.figure(figsize=(12, 8))
        for family in top_families:
            family_data = plot_data[plot_data['outbound_fare_family'] == family]
            if len(family_data) > 0:
                plt.scatter(family_data['umap_1'], family_data['umap_2'], 
                           c=[family_colors[family]], label=family, 
                           alpha=0.7, s=12, edgecolors='none')
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.title(f'UMAP: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'umap_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 4. UMAP by Cluster
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(plot_data['umap_1'], plot_data['umap_2'], 
                            c=plot_data['kmeans_cluster'], cmap='tab10', 
                            alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.title(f'UMAP: K-Means Clustering Results ({brand_name})', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'umap_clusters_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 5. t-SNE by Fare Family
        tsne_brand_data = self.tsne_data[self.tsne_data['brand'] == brand]
        if len(tsne_brand_data) > 0:
            plt.figure(figsize=(12, 8))
            tsne_top_families = tsne_brand_data['outbound_fare_family'].value_counts().head(10).index
            for family in tsne_top_families:
                family_data = tsne_brand_data[tsne_brand_data['outbound_fare_family'] == family]
                if len(family_data) > 0:
                    plt.scatter(family_data['tsne_1'], family_data['tsne_2'], 
                               c=[family_colors[family]], label=family, 
                               alpha=0.7, s=12, edgecolors='none')
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.title(f't-SNE: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'tsne_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # 6. t-SNE by Cluster
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(tsne_brand_data['tsne_1'], tsne_brand_data['tsne_2'], 
                                c=tsne_brand_data['kmeans_cluster'], cmap='tab10', 
                                alpha=0.6, s=8, edgecolors='none')
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.title(f't-SNE: K-Means Clustering Results ({brand_name})', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'tsne_clusters_{brand}.png', dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
        
        # 7. Fare Family Distribution
        brand_data = self.data[self.data['brand'] == brand]
        plt.figure(figsize=(14, 8))
        fare_family_counts = brand_data['outbound_fare_family'].value_counts()
        top_families_bar = fare_family_counts.head(12)
        bar_colors = [family_colors[family] for family in top_families_bar.index]
        
        bars = plt.bar(range(len(top_families_bar)), top_families_bar.values, 
                      color=bar_colors, alpha=0.8, width=0.8)
        plt.title(f'Top 12 Fare Family Distribution ({brand_name})', fontsize=14, fontweight='bold')
        plt.xlabel('Fare Family', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(range(len(top_families_bar)), top_families_bar.index, 
                  rotation=45, ha='right', fontsize=9)
        
        # Add value labels on bars
        for bar, value in zip(bars, top_families_bar.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{value:,}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'fare_family_distribution_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 8. Price vs Duration by Fare Family
        plt.figure(figsize=(12, 8))
        for family in top_families:
            family_data = plot_data[plot_data['outbound_fare_family'] == family]
            if len(family_data) > 0:
                plt.scatter(family_data['duration'], family_data['price_inc'], 
                           c=[family_colors[family]], label=family, 
                           alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('Duration (minutes)', fontsize=12)
        plt.ylabel('Price (including taxes)', fontsize=12)
        plt.title(f'Price vs Duration by Fare Family ({brand_name})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'price_duration_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 9. Kernel PCA (RBF) by Fare Family
        plt.figure(figsize=(12, 8))
        for family in top_families:
            family_data = plot_data[plot_data['outbound_fare_family'] == family]
            if len(family_data) > 0:
                plt.scatter(family_data['kpca_1'], family_data['kpca_2'], 
                           c=[family_colors[family]], label=family, 
                           alpha=0.7, s=12, edgecolors='none')
        plt.xlabel('Kernel PCA (RBF) Dimension 1', fontsize=12)
        plt.ylabel('Kernel PCA (RBF) Dimension 2', fontsize=12)
        plt.title(f'Kernel PCA (RBF): Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'kpca_rbf_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 10. Kernel PCA (Polynomial) by Fare Family
        plt.figure(figsize=(12, 8))
        for family in top_families:
            family_data = plot_data[plot_data['outbound_fare_family'] == family]
            if len(family_data) > 0:
                plt.scatter(family_data['kpca_poly_1'], family_data['kpca_poly_2'], 
                           c=[family_colors[family]], label=family, 
                           alpha=0.7, s=12, edgecolors='none')
        plt.xlabel('Kernel PCA (Poly) Dimension 1', fontsize=12)
        plt.ylabel('Kernel PCA (Poly) Dimension 2', fontsize=12)
        plt.title(f'Kernel PCA (Polynomial): Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'kpca_poly_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 11. Sparse PCA by Fare Family
        plt.figure(figsize=(12, 8))
        for family in top_families:
            family_data = plot_data[plot_data['outbound_fare_family'] == family]
            if len(family_data) > 0:
                plt.scatter(family_data['spca_1'], family_data['spca_2'], 
                           c=[family_colors[family]], label=family, 
                           alpha=0.7, s=12, edgecolors='none')
        plt.xlabel('Sparse PCA Dimension 1', fontsize=12)
        plt.ylabel('Sparse PCA Dimension 2', fontsize=12)
        plt.title(f'Sparse PCA: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'spca_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 12. Factor Analysis by Fare Family
        plt.figure(figsize=(12, 8))
        for family in top_families:
            family_data = plot_data[plot_data['outbound_fare_family'] == family]
            if len(family_data) > 0:
                plt.scatter(family_data['fa_1'], family_data['fa_2'], 
                           c=[family_colors[family]], label=family, 
                           alpha=0.7, s=12, edgecolors='none')
        plt.xlabel('Factor Analysis Dimension 1', fontsize=12)
        plt.ylabel('Factor Analysis Dimension 2', fontsize=12)
        plt.title(f'Factor Analysis: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'fa_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 13. MDS by Fare Family
        mds_brand_data = self.mds_data[self.mds_data['brand'] == brand]
        if len(mds_brand_data) > 0:
            plt.figure(figsize=(12, 8))
            mds_top_families = mds_brand_data['outbound_fare_family'].value_counts().head(10).index
            for family in mds_top_families:
                family_data = mds_brand_data[mds_brand_data['outbound_fare_family'] == family]
                if len(family_data) > 0:
                    plt.scatter(family_data['mds_1'], family_data['mds_2'], 
                               c=[family_colors[family]], label=family, 
                               alpha=0.7, s=12, edgecolors='none')
            plt.xlabel('MDS Dimension 1', fontsize=12)
            plt.ylabel('MDS Dimension 2', fontsize=12)
            plt.title(f'MDS: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'mds_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
        
        # 14. Isomap by Fare Family
        iso_brand_data = self.iso_data[self.iso_data['brand'] == brand]
        if len(iso_brand_data) > 0:
            plt.figure(figsize=(12, 8))
            iso_top_families = iso_brand_data['outbound_fare_family'].value_counts().head(10).index
            for family in iso_top_families:
                family_data = iso_brand_data[iso_brand_data['outbound_fare_family'] == family]
                if len(family_data) > 0:
                    plt.scatter(family_data['iso_1'], family_data['iso_2'], 
                               c=[family_colors[family]], label=family, 
                               alpha=0.7, s=12, edgecolors='none')
            plt.xlabel('Isomap Dimension 1', fontsize=12)
            plt.ylabel('Isomap Dimension 2', fontsize=12)
            plt.title(f'Isomap: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'isomap_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
        
        # 15. LLE by Fare Family (if available)
        if self.has_lle:
            lle_brand_data = self.lle_data[self.lle_data['brand'] == brand]
            if len(lle_brand_data) > 0:
                plt.figure(figsize=(12, 8))
                lle_top_families = lle_brand_data['outbound_fare_family'].value_counts().head(10).index
                for family in lle_top_families:
                    family_data = lle_brand_data[lle_brand_data['outbound_fare_family'] == family]
                    if len(family_data) > 0:
                        plt.scatter(family_data['lle_1'], family_data['lle_2'], 
                                   c=[family_colors[family]], label=family, 
                                   alpha=0.7, s=12, edgecolors='none')
                plt.xlabel('LLE Dimension 1', fontsize=12)
                plt.ylabel('LLE Dimension 2', fontsize=12)
                plt.title(f'LLE: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
                plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'lle_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
        
        # 16. LDA by Fare Family (if available)
        if self.has_lda:
            plt.figure(figsize=(12, 8))
            for family in top_families:
                family_data = plot_data[plot_data['outbound_fare_family'] == family]
                if len(family_data) > 0:
                    plt.scatter(family_data['lda_1'], family_data['lda_2'], 
                               c=[family_colors[family]], label=family, 
                               alpha=0.7, s=12, edgecolors='none')
            plt.xlabel('LDA Dimension 1', fontsize=12)
            plt.ylabel('LDA Dimension 2', fontsize=12)
            plt.title(f'LDA: Top 10 Fare Families ({brand_name})', fontsize=14, fontweight='bold')
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'lda_fare_families_{brand}.png', dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
        
        print(f"Individual subplots saved for {brand_name} (including advanced dimensionality reduction methods)")
        
    def create_combined_overview_plot(self):
        """Create combined overview plot for all airlines"""
        print("Creating combined overview plot...")
        
        # Set up the figure with optimized size
        fig = plt.figure(figsize=(18, 22))
        
        # Color palettes for airlines
        brand_colors = {'AA': '#FF6B6B', 'B6': '#4ECDC4', 'AS': '#45B7D1'}
        
        # Sample data for plotting efficiency
        plot_data = self.data.sample(n=min(15000, len(self.data)), random_state=42) if len(self.data) > 15000 else self.data
        
        # 1. PCA Visualization by Airline Brand
        plt.subplot(4, 2, 1)
        for brand in plot_data['brand'].unique():
            brand_data = plot_data[plot_data['brand'] == brand]
            plt.scatter(brand_data['pca_1'], brand_data['pca_2'], 
                       c=brand_colors[brand], label=self.brand_mapping[brand], 
                       alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.title('PCA: Fare Family Distribution by Airline Brand', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 2. PCA Visualization by Cluster
        plt.subplot(4, 2, 2)
        scatter = plt.scatter(plot_data['pca_1'], plot_data['pca_2'], 
                            c=plot_data['kmeans_cluster'], cmap='tab10', 
                            alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.title('PCA: K-Means Clustering Results (All Airlines)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        # 3. UMAP Visualization by Airline Brand
        plt.subplot(4, 2, 3)
        for brand in plot_data['brand'].unique():
            brand_data = plot_data[plot_data['brand'] == brand]
            plt.scatter(brand_data['umap_1'], brand_data['umap_2'], 
                       c=brand_colors[brand], label=self.brand_mapping[brand], 
                       alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.title('UMAP: Fare Family Distribution by Airline Brand', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 4. UMAP Visualization by Cluster
        plt.subplot(4, 2, 4)
        scatter = plt.scatter(plot_data['umap_1'], plot_data['umap_2'], 
                            c=plot_data['kmeans_cluster'], cmap='tab10', 
                            alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.title('UMAP: K-Means Clustering Results (All Airlines)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        # 5. t-SNE Visualization by Airline Brand
        plt.subplot(4, 2, 5)
        for brand in self.tsne_data['brand'].unique():
            brand_data = self.tsne_data[self.tsne_data['brand'] == brand]
            plt.scatter(brand_data['tsne_1'], brand_data['tsne_2'], 
                       c=brand_colors[brand], label=self.brand_mapping[brand], 
                       alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE: Fare Family Distribution by Airline Brand', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 6. t-SNE Visualization by Cluster
        plt.subplot(4, 2, 6)
        scatter = plt.scatter(self.tsne_data['tsne_1'], self.tsne_data['tsne_2'], 
                            c=self.tsne_data['kmeans_cluster'], cmap='tab10', 
                            alpha=0.6, s=8, edgecolors='none')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE: K-Means Clustering Results (All Airlines)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        # 7. Fare Family Distribution by Airline
        plt.subplot(4, 2, 7)
        fare_family_counts = self.data.groupby(['brand_name', 'outbound_fare_family']).size().reset_index(name='count')
        pivot_data = fare_family_counts.pivot(index='outbound_fare_family', columns='brand_name', values='count').fillna(0)
        
        # Select top fare families for readability
        top_families = pivot_data.sum(axis=1).nlargest(8).index
        pivot_subset = pivot_data.loc[top_families]
        
        pivot_subset.plot(kind='bar', stacked=True, ax=plt.gca(), width=0.8)
        plt.title('Top 8 Fare Family Distribution by Airline', fontsize=14, fontweight='bold')
        plt.xlabel('Fare Family', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.legend(title='Airline', fontsize=10)
        
        # 8. Price vs Duration by Airline Brand
        plt.subplot(4, 2, 8)
        for brand in plot_data['brand'].unique():
            brand_data = plot_data[plot_data['brand'] == brand]
            plt.scatter(brand_data['duration'], brand_data['price_inc'], 
                       c=brand_colors[brand], label=self.brand_mapping[brand], 
                       alpha=0.5, s=6, edgecolors='none')
        plt.xlabel('Duration (minutes)', fontsize=12)
        plt.ylabel('Price (including taxes)', fontsize=12)
        plt.title('Price vs Duration by Airline Brand', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0)
        plt.savefig('fare_family_clustering_analysis_ALL.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()  # Close to free memory
        print("Combined visualization saved as 'fare_family_clustering_analysis_ALL.png'")
        
    def print_analysis_summary(self):
        """Print summary of findings"""
        print("\n" + "="*60)
        print("MULTI-AIRLINE FARE FAMILY CLUSTERING ANALYSIS")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"- Total records analyzed: {len(self.data):,}")
        print(f"- Airlines: {', '.join([self.brand_mapping[b] for b in self.data['brand'].unique()])}")
        print(f"- Unique fare families: {self.data['outbound_fare_family'].nunique()}")
        
        print(f"\nFare Family Distribution:")
        for brand in self.data['brand'].unique():
            brand_data = self.data[self.data['brand'] == brand]
            print(f"- {self.brand_mapping[brand]}: {len(brand_data):,} records, "
                  f"{brand_data['outbound_fare_family'].nunique()} unique fare families")
        
        print(f"\nTop 5 Most Common Fare Families:")
        top_families = self.data['outbound_fare_family'].value_counts().head()
        for family, count in top_families.items():
            print(f"- {family}: {count:,} records")
        
        print(f"\nClustering Results:")
        
        print("Cluster characteristics:")
        for cluster in sorted(self.data['kmeans_cluster'].unique()):
            cluster_data = self.data[self.data['kmeans_cluster'] == cluster]
            print(f"- Cluster {cluster}: {len(cluster_data):,} records, "
                  f"avg price ${cluster_data['price_inc'].mean():.2f}, "
                  f"avg duration {cluster_data['duration'].mean():.0f}min, "
                  f"{cluster_data['outbound_fare_family'].nunique()} fare families")


def main():
    """Main execution function"""
    # Initialize visualizer
    visualizer = FareFamilyClusteringVisualizer()
    
    # File paths - All airlines
    file_paths = [
        '/Users/weichengzeng/Library/CloudStorage/OneDrive-ATPCO/Desktop/visualize/brands_AA.csv',
        '/Users/weichengzeng/Library/CloudStorage/OneDrive-ATPCO/Desktop/visualize/brands_B6.csv',
        '/Users/weichengzeng/Library/CloudStorage/OneDrive-ATPCO/Desktop/visualize/brands_AS.csv'
    ]
    
    # Execute analysis pipeline
    visualizer.load_and_preprocess_data(file_paths)
    visualizer.perform_clustering()
    visualizer.apply_dimensionality_reduction()
    visualizer.create_visualizations()
    visualizer.print_analysis_summary()
    
    print(f"\nAnalysis complete! Individual and combined visualizations saved.")


if __name__ == "__main__":
    main()
