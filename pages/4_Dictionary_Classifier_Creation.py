import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Instagram Dictionary Classifier",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DictionaryClassifier:
    def __init__(self):
        self.df = None
        self.engagement_df = None
        self.keywords = []
        self.feature_matrix = None
        self.target_variable = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_name = None
        
    def load_data_from_uploads(self, posts_df, engagement_df=None, keywords_list=None):
        """Load data from Streamlit uploads"""
        try:
            self.df = posts_df.copy()
            
            required_cols = ['ID', 'Statement']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                st.error(f"Missing required columns in posts data: {missing_cols}")
                return False
                
            # Load engagement data
            if engagement_df is not None:
                self.engagement_df = engagement_df.copy()
                
                # Merge with posts data
                self.df = pd.merge(self.df, self.engagement_df, 
                                 left_on='ID', right_on='shortcode', how='inner')
                st.success(f"✅ Merged data: {self.df.shape[0]} posts with engagement metrics")
            else:
                st.warning("⚠️ No engagement data provided - cannot create engagement-based classifier")
                return False
                
            # Load keywords
            if keywords_list:
                self.keywords = [kw.strip().lower() for kw in keywords_list if kw.strip()]
                st.success(f"✅ Loaded {len(self.keywords)} keywords")
            else:
                self.keywords = self._get_default_keywords()
                st.info(f"📝 Using {len(self.keywords)} default keywords")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def _get_default_keywords(self):
        """Return default keyword list"""
        return [
            "personalized", "custom", "tailored", "bespoke", "individual",
            "human", "experience", "connection", "care", "thoughtful", 
            "responsive", "concierge", "attentive", "dedicated", "unique",
            "special", "exclusive", "premium", "luxury", "curated",
            "handpicked", "artisan", "boutique", "intimate", "personal",
            "authentic", "genuine", "original", "distinctive", "exceptional",
            "remarkable", "outstanding", "extraordinary", "memorable", "unforgettable",
            "amazing", "incredible", "fantastic", "wonderful", "beautiful",
            "stunning", "gorgeous", "elegant", "sophisticated", "refined"
        ]
    
    def create_features(self, min_word_count=5):
        """Create feature matrix from keywords"""
        features = []
        valid_posts = []
        
        # Keyword categories for additional features
        emotion_keywords = ["amazing", "incredible", "fantastic", "wonderful", "beautiful", "stunning", "gorgeous"]
        quality_keywords = ["premium", "luxury", "exclusive", "exceptional", "outstanding", "extraordinary"]
        personal_keywords = ["personalized", "custom", "tailored", "bespoke", "individual", "personal"]
        service_keywords = ["care", "thoughtful", "responsive", "concierge", "attentive", "dedicated"]
        
        for idx, row in self.df.iterrows():
            if pd.isna(row['Statement']):
                continue
                
            text = str(row['Statement']).lower()
            words = re.findall(r'\b\w+\b', text)
            
            # Skip posts that are too short
            if len(words) < min_word_count:
                continue
                
            # Basic features
            post_features = {
                'total_words': len(words),
                'total_keywords': 0,
                'keyword_density': 0,
                'unique_keywords': 0
            }
            
            # Individual keyword presence (binary features)
            matched_keywords = set()
            for keyword in self.keywords:
                if keyword in words:
                    post_features[f'has_{keyword}'] = 1
                    matched_keywords.add(keyword)
                    post_features['total_keywords'] += words.count(keyword)
                else:
                    post_features[f'has_{keyword}'] = 0
            
            # Derived features
            post_features['unique_keywords'] = len(matched_keywords)
            post_features['keyword_density'] = post_features['total_keywords'] / len(words) if words else 0
            
            # Category-based features
            post_features['emotion_score'] = sum(1 for kw in emotion_keywords if kw in matched_keywords)
            post_features['quality_score'] = sum(1 for kw in quality_keywords if kw in matched_keywords)
            post_features['personal_score'] = sum(1 for kw in personal_keywords if kw in matched_keywords)
            post_features['service_score'] = sum(1 for kw in service_keywords if kw in matched_keywords)
            
            # Text-based features
            post_features['has_exclamation'] = 1 if '!' in row['Statement'] else 0
            post_features['has_question'] = 1 if '?' in row['Statement'] else 0
            post_features['has_hashtag'] = 1 if '#' in row['Statement'] else 0
            post_features['has_mention'] = 1 if '@' in row['Statement'] else 0
            post_features['char_count'] = len(row['Statement'])
            post_features['sentence_count'] = len(re.split(r'[.!?]+', row['Statement']))
            
            features.append(post_features)
            valid_posts.append(idx)
        
        # Convert to DataFrame
        self.feature_matrix = pd.DataFrame(features)
        self.df = self.df.iloc[valid_posts].reset_index(drop=True)
        
        return self.feature_matrix
    
    def create_target_variables(self, engagement_threshold_percentile=75):
        """Create target variables for classification"""
        if 'number_likes' not in self.df.columns or 'number_comments' not in self.df.columns:
            st.error("❌ Engagement data not available for target creation")
            return False
        
        # Calculate engagement metrics
        self.df['total_engagement'] = self.df['number_likes'] + self.df['number_comments']
        self.df['engagement_rate'] = self.df['total_engagement'] / self.df['number_likes'].max()  # Normalized
        
        # Create binary targets based on percentiles
        likes_threshold = np.percentile(self.df['number_likes'], engagement_threshold_percentile)
        comments_threshold = np.percentile(self.df['number_comments'], engagement_threshold_percentile)
        engagement_threshold = np.percentile(self.df['total_engagement'], engagement_threshold_percentile)
        
        self.df['high_likes'] = (self.df['number_likes'] >= likes_threshold).astype(int)
        self.df['high_comments'] = (self.df['number_comments'] >= comments_threshold).astype(int)
        self.df['high_engagement'] = (self.df['total_engagement'] >= engagement_threshold).astype(int)
        
        # Multi-class target (low, medium, high engagement)
        engagement_33 = np.percentile(self.df['total_engagement'], 33)
        engagement_66 = np.percentile(self.df['total_engagement'], 66)
        
        def categorize_engagement(value):
            if value <= engagement_33:
                return 0  # Low
            elif value <= engagement_66:
                return 1  # Medium
            else:
                return 2  # High
        
        self.df['engagement_category'] = self.df['total_engagement'].apply(categorize_engagement)
        
        return True
    
    def train_classifiers(self, target='high_engagement', test_size=0.2, use_feature_selection=True):
        """Train multiple classification models"""
        if target not in self.df.columns:
            st.error(f"❌ Target variable '{target}' not found")
            return False
        
        # Prepare data
        X = self.feature_matrix.copy()
        y = self.df[target].copy()
        
        # Feature selection
        if use_feature_selection and X.shape[1] > 20:
            k_features = min(20, X.shape[1] // 2)  # Select top features
            self.feature_selector = SelectKBest(score_func=f_classif, k=k_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            X = pd.DataFrame(X_selected, columns=selected_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                           random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
        }
        
        # Train and evaluate models
        results = {}
        progress_bar = st.progress(0)
        
        for i, (name, model) in enumerate(models_config.items()):
            st.info(f"🔧 Training {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            # Train model
            model.fit(X_train_use, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_use)
            y_prob = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = model.score(X_test_use, y_test)
            
            if y_prob is not None:
                auc_score = roc_auc_score(y_test, y_prob)
            else:
                auc_score = None
            
            # Cross-validation
            if name in ['SVM', 'Logistic Regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_prob': y_prob,
                'y_test': y_test
            }
            
            progress_bar.progress((i + 1) / len(models_config))
        
        self.models[target] = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        # Store training data for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_name = target
        
        return results
    
    def predict_engagement(self, text_samples, target='high_engagement'):
        """Predict engagement for new text samples"""
        if target not in self.models or self.best_model is None:
            st.error(f"❌ No trained model available for target: {target}")
            return None, None
        
        # Create features for new samples
        predictions = []
        
        for text in text_samples:
            # Create feature vector
            words = re.findall(r'\b\w+\b', text.lower())
            
            features = {
                'total_words': len(words),
                'total_keywords': 0,
                'keyword_density': 0,
                'unique_keywords': 0
            }
            
            # Individual keyword presence
            matched_keywords = set()
            for keyword in self.keywords:
                if keyword in words:
                    features[f'has_{keyword}'] = 1
                    matched_keywords.add(keyword)
                    features['total_keywords'] += words.count(keyword)
                else:
                    features[f'has_{keyword}'] = 0
            
            # Derived features
            features['unique_keywords'] = len(matched_keywords)
            features['keyword_density'] = features['total_keywords'] / len(words) if words else 0
            
            # Category features
            emotion_keywords = ["amazing", "incredible", "fantastic", "wonderful", "beautiful", "stunning", "gorgeous"]
            quality_keywords = ["premium", "luxury", "exclusive", "exceptional", "outstanding", "extraordinary"]
            personal_keywords = ["personalized", "custom", "tailored", "bespoke", "individual", "personal"]
            service_keywords = ["care", "thoughtful", "responsive", "concierge", "attentive", "dedicated"]
            
            features['emotion_score'] = sum(1 for kw in emotion_keywords if kw in matched_keywords)
            features['quality_score'] = sum(1 for kw in quality_keywords if kw in matched_keywords)
            features['personal_score'] = sum(1 for kw in personal_keywords if kw in matched_keywords)
            features['service_score'] = sum(1 for kw in service_keywords if kw in matched_keywords)
            
            # Text features
            features['has_exclamation'] = 1 if '!' in text else 0
            features['has_question'] = 1 if '?' in text else 0
            features['has_hashtag'] = 1 if '#' in text else 0
            features['has_mention'] = 1 if '@' in text else 0
            features['char_count'] = len(text)
            features['sentence_count'] = len(re.split(r'[.!?]+', text))
            
            predictions.append(features)
        
        # Convert to DataFrame and align with training features
        pred_df = pd.DataFrame(predictions)
        
        # Ensure all training features are present
        for col in self.X_train.columns:
            if col not in pred_df.columns:
                pred_df[col] = 0
        
        # Reorder columns to match training data
        pred_df = pred_df[self.X_train.columns]
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            pred_df = pd.DataFrame(
                self.feature_selector.transform(pred_df),
                columns=self.X_train.columns[self.feature_selector.get_support()]
            )
        
        # Scale if needed (for SVM/LogReg)
        best_model_name = max(self.models[target].keys(), 
                             key=lambda x: self.models[target][x]['cv_mean'])
        
        if best_model_name in ['SVM', 'Logistic Regression']:
            pred_features = self.scaler.transform(pred_df)
        else:
            pred_features = pred_df
        
        # Make predictions
        predictions = self.best_model.predict(pred_features)
        probabilities = self.best_model.predict_proba(pred_features)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        return predictions, probabilities

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = DictionaryClassifier()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'features_created' not in st.session_state:
    st.session_state.features_created = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Main app
def main():
    st.markdown("<h1 class='main-header'>📊 Instagram Dictionary Classifier</h1>", unsafe_allow_html=True)
    st.markdown("### Create ML models to predict Instagram engagement based on personalized language patterns")
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📁 Data Upload", "🔧 Feature Engineering", "🤖 Model Training", "📊 Model Evaluation", "🔮 Predictions", "💾 Export Model"]
    )
    
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "🔧 Feature Engineering":
        feature_engineering_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "📊 Model Evaluation":
        model_evaluation_page()
    elif page == "🔮 Predictions":
        predictions_page()
    elif page == "💾 Export Model":
        export_page()

def data_upload_page():
    st.markdown("<h2 class='section-header'>📁 Data Upload</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload your Instagram data files to get started. You need:
    1. **Posts Data**: CSV with columns 'ID' and 'Statement'
    2. **Engagement Data**: CSV with columns 'shortcode', 'number_likes', 'number_comments'
    3. **Keywords** (optional): Text file or manual input
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Posts Data")
        posts_file = st.file_uploader("Upload posts CSV", type=['csv'], key="posts")
        
        if posts_file:
            try:
                posts_df = pd.read_csv(posts_file)
                st.success(f"✅ Loaded {len(posts_df)} posts")
                st.dataframe(posts_df.head())
                
                # Check required columns
                required_cols = ['ID', 'Statement']
                missing_cols = [col for col in required_cols if col not in posts_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.posts_df = posts_df
                    
            except Exception as e:
                st.error(f"Error loading posts file: {str(e)}")
    
    with col2:
        st.subheader("📈 Engagement Data")
        engagement_file = st.file_uploader("Upload engagement CSV", type=['csv'], key="engagement")
        
        if engagement_file:
            try:
                engagement_df = pd.read_csv(engagement_file)
                st.success(f"✅ Loaded {len(engagement_df)} engagement records")
                st.dataframe(engagement_df.head())
                
                # Check required columns
                required_cols = ['shortcode', 'number_likes', 'number_comments']
                missing_cols = [col for col in required_cols if col not in engagement_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.engagement_df = engagement_df
                    
            except Exception as e:
                st.error(f"Error loading engagement file: {str(e)}")
    
    # Keywords section
    st.subheader("🔑 Keywords")
    
    keyword_option = st.radio(
        "Choose keyword source:",
        ["Use default keywords", "Upload keywords file", "Enter keywords manually"]
    )
    
    keywords_list = None
    
    if keyword_option == "Upload keywords file":
        keywords_file = st.file_uploader("Upload keywords file", type=['txt', 'csv'], key="keywords")
        if keywords_file:
            try:
                if keywords_file.name.endswith('.txt'):
                    keywords_list = keywords_file.read().decode('utf-8').split('\n')
                else:
                    kw_df = pd.read_csv(keywords_file)
                    keywords_list = kw_df.iloc[:, 0].tolist()
                st.success(f"✅ Loaded {len(keywords_list)} keywords")
            except Exception as e:
                st.error(f"Error loading keywords file: {str(e)}")
                
    elif keyword_option == "Enter keywords manually":
        keywords_text = st.text_area(
            "Enter keywords (one per line):",
            height=200,
            placeholder="personalized\ncustom\ntailored\nbespoke\n..."
        )
        if keywords_text:
            keywords_list = [kw.strip() for kw in keywords_text.split('\n') if kw.strip()]
            st.info(f"📝 {len(keywords_list)} keywords entered")
    
    # Load data button
    if st.button("🚀 Load Data", type="primary"):
        if 'posts_df' in st.session_state and 'engagement_df' in st.session_state:
            success = st.session_state.classifier.load_data_from_uploads(
                st.session_state.posts_df,
                st.session_state.engagement_df,
                keywords_list
            )
            if success:
                st.session_state.data_loaded = True
                st.success("🎉 Data loaded successfully! Go to Feature Engineering.")
        else:
            st.error("❌ Please upload both posts and engagement data files.")

def feature_engineering_page():
    st.markdown("<h2 class='section-header'>🔧 Feature Engineering</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Upload section.")
        return
    
    st.markdown("Configure feature creation parameters:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_word_count = st.slider("Minimum word count per post", 1, 20, 5)
        engagement_threshold = st.slider("High engagement threshold (percentile)", 50, 95, 75)
    
    with col2:
        st.info(f"""
        **Current Data:**
        - Posts: {len(st.session_state.classifier.df)}
        - Keywords: {len(st.session_state.classifier.keywords)}
        """)
    
    if st.button("🔧 Create Features", type="primary"):
        with st.spinner("Creating features..."):
            # Create features
            feature_matrix = st.session_state.classifier.create_features(min_word_count)
            
            # Create target variables
            success = st.session_state.classifier.create_target_variables(engagement_threshold)
            
            if success:
                st.session_state.features_created = True
                
                st.success(f"✅ Created {feature_matrix.shape[1]} features for {feature_matrix.shape[0]} posts")
                
                # Display feature summary
                st.subheader("📊 Feature Summary")
                
                # Basic stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_keywords = feature_matrix['total_keywords'].mean()
                    st.metric("Avg Keywords per Post", f"{avg_keywords:.1f}")
                
                with col2:
                    avg_density = feature_matrix['keyword_density'].mean()
                    st.metric("Avg Keyword Density", f"{avg_density:.3f}")
                
                with col3:
                    high_engagement_posts = st.session_state.classifier.df['high_engagement'].sum()
                    st.metric("High Engagement Posts", high_engagement_posts)
                
                with col4:
                    total_posts = len(st.session_state.classifier.df)
                    engagement_rate = (high_engagement_posts / total_posts) * 100
                    st.metric("High Engagement Rate", f"{engagement_rate:.1f}%")
                
                # Feature correlation analysis
                st.subheader("🔍 Top Feature Correlations")
                
                correlations = []
                for col in feature_matrix.columns[:20]:  # Top 20 features
                    corr = feature_matrix[col].corr(st.session_state.classifier.df['high_engagement'])
                    if not pd.isna(corr):
                        correlations.append((col, abs(corr), corr))
                
                correlations.sort(key=lambda x: x[1], reverse=True)
                
                # Display top correlations
                corr_df = pd.DataFrame(correlations[:10], columns=['Feature', 'Abs_Correlation', 'Correlation'])
                st.dataframe(corr_df)
                
                st.success("🎉 Features created successfully! Go to Model Training.")

def model_evaluation_page():
    st.markdown("<h2 class='section-header'>📊 Model Evaluation</h2>", unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training section.")
        return
    
    target = st.session_state.current_target
    results = st.session_state.classifier.models[target]
    
    # Model selection for detailed analysis
    model_names = list(results.keys())
    selected_model = st.selectbox("Select model for detailed analysis:", model_names)
    
    if selected_model:
        model_data = results[selected_model]
        
        # Performance metrics
        st.subheader(f"📈 {selected_model} Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test Accuracy", f"{model_data['accuracy']:.4f}")
        
        with col2:
            st.metric("CV Accuracy", f"{model_data['cv_mean']:.4f}")
        
        with col3:
            st.metric("CV Std Dev", f"{model_data['cv_std']:.4f}")
        
        with col4:
            if model_data['auc_score']:
                st.metric("AUC Score", f"{model_data['auc_score']:.4f}")
            else:
                st.metric("AUC Score", "N/A")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Confusion Matrix")
            cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("📊 Classification Report")
            report = classification_report(model_data['y_test'], model_data['y_pred'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4))
        
        # ROC Curve (if probabilities available)
        if model_data['y_prob'] is not None:
            st.subheader("📈 ROC Curve")
            fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_prob'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'{selected_model} (AUC = {model_data["auc_score"]:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close()
        
        # Feature Importance (for tree-based models)
        if hasattr(model_data['model'], 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            feature_names = st.session_state.classifier.X_train.columns
            importances = model_data['model'].feature_importances_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot top 15 features
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = importance_df.head(15)
            sns.barplot(data=top_features, x='Importance', y='Feature', ax=ax)
            ax.set_title(f'Top 15 Feature Importances - {selected_model}')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show feature importance table
            st.dataframe(importance_df.head(20))

def predictions_page():
    st.markdown("<h2 class='section-header'>🔮 Predictions</h2>", unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training section.")
        return
    
    st.markdown("### Test your model on new Instagram post text")
    
    # Single prediction
    st.subheader("📝 Single Post Prediction")
    
    sample_text = st.text_area(
        "Enter Instagram post text:",
        height=150,
        placeholder="Enter your Instagram post text here to predict its engagement potential..."
    )
    
    if sample_text and st.button("🔍 Predict Engagement", type="primary"):
        target = st.session_state.current_target
        predictions, probabilities = st.session_state.classifier.predict_engagement([sample_text], target)
        
        if predictions is not None:
            prediction = predictions[0]
            probability = probabilities[0] if probabilities is not None else None
            
            col1, col2 = st.columns(2)
            
            with col1:
                if target == 'engagement_category':
                    categories = ['Low', 'Medium', 'High']
                    result = categories[prediction]
                    st.success(f"**Predicted Engagement Level: {result}**")
                else:
                    result = "High Engagement" if prediction == 1 else "Low Engagement"
                    st.success(f"**Prediction: {result}**")
            
            with col2:
                if probability is not None:
                    confidence = probability if prediction == 1 else (1 - probability)
                    st.info(f"**Confidence: {confidence:.2%}**")
            
            # Feature analysis
            st.subheader("🔍 Text Analysis")
            
            # Extract features for analysis
            text = sample_text.lower()
            words = re.findall(r'\b\w+\b', text)
            matched_keywords = [kw for kw in st.session_state.classifier.keywords if kw in words]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Word Count", len(words))
                st.metric("Character Count", len(sample_text))
            
            with col2:
                st.metric("Keywords Found", len(matched_keywords))
                keyword_density = len(matched_keywords) / len(words) if words else 0
                st.metric("Keyword Density", f"{keyword_density:.3f}")
            
            with col3:
                has_exclamation = "Yes" if "!" in sample_text else "No"
                st.metric("Has Exclamation", has_exclamation)
                has_hashtag = "Yes" if "#" in sample_text else "No"
                st.metric("Has Hashtag", has_hashtag)
            
            if matched_keywords:
                st.subheader("✅ Keywords Found")
                st.write(", ".join(matched_keywords))
    
    # Batch prediction
    st.subheader("📊 Batch Prediction")
    st.markdown("Upload a CSV file with a 'text' column to predict engagement for multiple posts.")
    
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
    
    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
            
            if 'text' not in batch_df.columns:
                st.error("❌ CSV must contain a 'text' column")
            else:
                st.success(f"✅ Loaded {len(batch_df)} posts for prediction")
                st.dataframe(batch_df.head())
                
                if st.button("🚀 Run Batch Prediction"):
                    with st.spinner("Making predictions..."):
                        target = st.session_state.current_target
                        predictions, probabilities = st.session_state.classifier.predict_engagement(
                            batch_df['text'].tolist(), target
                        )
                        
                        if predictions is not None:
                            # Add predictions to dataframe
                            if target == 'engagement_category':
                                categories = ['Low', 'Medium', 'High']
                                batch_df['predicted_engagement'] = [categories[p] for p in predictions]
                            else:
                                batch_df['predicted_engagement'] = ['High' if p == 1 else 'Low' for p in predictions]
                            
                            if probabilities is not None:
                                batch_df['confidence'] = [p if pred == 1 else (1-p) for pred, p in zip(predictions, probabilities)]
                            
                            st.success("✅ Predictions completed!")
                            st.dataframe(batch_df)
                            
                            # Download results
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="engagement_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.subheader("📈 Prediction Summary")
                            
                            if target == 'engagement_category':
                                summary = batch_df['predicted_engagement'].value_counts()
                            else:
                                summary = batch_df['predicted_engagement'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Prediction Distribution:**")
                                st.dataframe(summary)
                            
                            with col2:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                summary.plot(kind='bar', ax=ax)
                                ax.set_title('Prediction Distribution')
                                ax.set_ylabel('Count')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def export_page():
    st.markdown("<h2 class='section-header'>💾 Export Model</h2>", unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training section.")
        return
    
    st.markdown("### Export your trained model and components")
    
    target = st.session_state.current_target
    
    # Model summary
    st.subheader("📋 Model Summary")
    
    results = st.session_state.classifier.models[target]
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model = results[best_model_name]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Model Details:**
        - Target Variable: {target}
        - Best Model: {best_model_name}
        - CV Accuracy: {best_model['cv_mean']:.4f}
        - Test Accuracy: {best_model['accuracy']:.4f}
        """)
    
    with col2:
        st.info(f"""
        **Data Details:**
        - Training Posts: {len(st.session_state.classifier.X_train)}
        - Features: {len(st.session_state.classifier.X_train.columns)}
        - Keywords: {len(st.session_state.classifier.keywords)}
        """)
    
    # Export options
    st.subheader("📦 Export Options")
    
    if st.button("💾 Download Model Package", type="primary"):
        # Create a dictionary with all necessary components
        model_package = {
            'best_model': st.session_state.classifier.best_model,
            'scaler': st.session_state.classifier.scaler,
            'feature_selector': st.session_state.classifier.feature_selector,
            'keywords': st.session_state.classifier.keywords,
            'feature_columns': st.session_state.classifier.X_train.columns.tolist(),
            'target_name': target,
            'model_name': best_model_name,
            'performance_metrics': {
                'cv_accuracy': best_model['cv_mean'],
                'test_accuracy': best_model['accuracy'],
                'auc_score': best_model['auc_score']
            }
        }
        
        # Serialize the model package
        buffer = io.BytesIO()
        pickle.dump(model_package, buffer)
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download Model Package (.pkl)",
            data=buffer.getvalue(),
            file_name=f"instagram_classifier_{target}_{best_model_name.lower().replace(' ', '_')}.pkl",
            mime="application/octet-stream"
        )
        
        st.success("✅ Model package ready for download!")
    
    # Usage instructions
    st.subheader("📖 Usage Instructions")
    
    st.markdown("""
    ### How to use the exported model:
    
    ```python
    import pickle
    import pandas as pd
    import numpy as np
    import re
    
    # Load the model package
    with open('your_model_file.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    # Extract components
    model = model_package['best_model']
    scaler = model_package['scaler']
    feature_selector = model_package['feature_selector']
    keywords = model_package['keywords']
    feature_columns = model_package['feature_columns']
    
    # Function to predict new text
    def predict_engagement(text):
        # Create features (same logic as in the app)
        words = re.findall(r'\\b\\w+\\b', text.lower())
        
        # ... feature creation logic ...
        # (Copy the feature creation code from the predict_engagement method)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1]
        
        return prediction[0], probability[0]
    
    # Example usage
    result = predict_engagement("Your Instagram post text here")
    print(f"Prediction: {result[0]}, Confidence: {result[1]:.2%}")
    ```
    """)
    
    # Model comparison export
    st.subheader("📊 Model Comparison Export")
    
    if st.button("📈 Export Model Comparison"):
        comparison_data = []
        
        for name, metrics in results.items():
            comparison_data.append({
                'Model': name,
                'CV_Accuracy': metrics['cv_mean'],
                'CV_Std': metrics['cv_std'],
                'Test_Accuracy': metrics['accuracy'],
                'AUC_Score': metrics['auc_score'] if metrics['auc_score'] else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        csv = comparison_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Model Comparison (CSV)",
            data=csv,
            file_name=f"model_comparison_{target}.csv",
            mime="text/csv"
        )

# Run the main app
if __name__ == "__main__":
    main()_training_page():
    st.markdown("<h2 class='section-header'>🤖 Model Training</h2>", unsafe_allow_html=True)
    
    if not st.session_state.features_created:
        st.warning("⚠️ Please create features first in the Feature Engineering section.")
        return
    
    st.markdown("Configure model training parameters:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_variable = st.selectbox(
            "Target Variable:",
            ["high_engagement", "high_likes", "high_comments", "engagement_category"]
        )
        
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        use_feature_selection = st.checkbox("Use feature selection", value=True)
        
        st.info(f"""
        **Training Configuration:**
        - Target: {target_variable}
        - Test size: {test_size*100:.0f}%
        - Feature selection: {'Yes' if use_feature_selection else 'No'}
        """)
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training multiple models..."):
            results = st.session_state.classifier.train_classifiers(
                target=target_variable,
                test_size=test_size,
                use_feature_selection=use_feature_selection
            )
            
            if results:
                st.session_state.models_trained = True
                st.session_state.current_target = target_variable
                
                st.success("🎉 Models trained successfully!")
                
                # Display results
                st.subheader("📊 Model Performance")
                
                results_data = []
                for name, metrics in results.items():
                    results_data.append({
                        'Model': name,
                        'Test Accuracy': f"{metrics['accuracy']:.4f}",
                        'CV Accuracy': f"{metrics['cv_mean']:.4f}",
                        'CV Std': f"{metrics['cv_std']:.4f}",
                        'AUC Score': f"{metrics['auc_score']:.4f}" if metrics['auc_score'] else "N/A"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df)
                
                # Best model
                best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
                best_score = results[best_model_name]['cv_mean']
                
                st.success(f"🏆 Best Model: **{best_model_name}** (CV Score: {best_score:.4f})")
                
                st.info("Go to Model Evaluation to see detailed analysis and visualizations.")

def model
