import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from collections import Counter, defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Instagram Category Classifier",
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CategoryClassifier:
    def __init__(self):
        self.df = None
        self.categories = {}
        self.classification_results = None
        self.keyword_frequency = {}
        self.ground_truth_available = False
        self.metrics = {}
        
    def load_data_from_uploads(self, posts_df, categories_dict=None, ground_truth_df=None):
        """Load data from Streamlit uploads"""
        try:
            self.df = posts_df.copy()
            
            required_cols = ['ID', 'Statement']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                st.error(f"Missing required columns in posts data: {missing_cols}")
                return False
            
            # Check for ground truth data
            if ground_truth_df is not None:
                if 'ID' in ground_truth_df.columns and 'True_Category' in ground_truth_df.columns:
                    # Merge ground truth with posts data
                    self.df = self.df.merge(ground_truth_df[['ID', 'True_Category']], on='ID', how='left')
                    self.ground_truth_available = True
                    st.success(f"✅ Ground truth data loaded for {len(ground_truth_df)} posts")
                else:
                    st.warning("⚠️ Ground truth file must have 'ID' and 'True_Category' columns")
            
            # Load categories
            if categories_dict:
                self.categories = categories_dict
                st.success(f"✅ Loaded {len(self.categories)} categories")
            else:
                self.categories = self._get_default_categories()
                st.info(f"📝 Using {len(self.categories)} default categories")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def _get_default_categories(self):
        """Return default category definitions"""
        return {
            "Gratitude Reflection": [
                "grateful", "thank", "thanks", "thankful", "blessed", "appreciate", 
                "appreciation", "gratitude", "honored", "privilege", "blessing"
            ],
            "Discount Pricing": [
                "discount", "sale", "off", "save", "savings", "deal", "offer", 
                "special price", "reduced", "markdown", "clearance", "promo"
            ],
            "Personal Milestone": [
                "milestone", "achievement", "accomplish", "goal", "journey", 
                "progress", "celebration", "success", "proud", "growth"
            ],
            "Social Proof": [
                "review", "testimonial", "customer", "client", "feedback", 
                "rating", "recommendation", "satisfied", "happy customer"
            ],
            "Urgency Marketing": [
                "limited", "hurry", "last chance", "ending soon", "don't miss", 
                "act fast", "while supplies last", "deadline", "urgent"
            ],
            "Exclusive Marketing": [
                "exclusive", "vip", "member", "special access", "invitation only", 
                "premium", "select", "insider", "private"
            ],
            "Local Business": [
                "local", "community", "neighborhood", "hometown", "area", 
                "nearby", "regional", "city", "town"
            ]
        }
    
    def classify_posts(self):
        """Classify posts into categories based on keywords"""
        if self.df is None or not self.categories:
            return None
            
        results = []
        category_counts = defaultdict(int)
        keyword_counts = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            if pd.isna(row['Statement']):
                continue
                
            text = str(row['Statement']).lower()
            words = re.findall(r'\b\w+\b', text)
            
            # Find matching categories
            post_categories = []
            found_keywords = []
            
            for category, keywords in self.categories.items():
                category_matches = []
                for keyword in keywords:
                    if keyword.lower() in text:
                        category_matches.append(keyword)
                        found_keywords.append(keyword)
                        keyword_counts[keyword] += 1
                
                if category_matches:
                    post_categories.append({
                        'category': category,
                        'keywords': category_matches,
                        'keyword_count': len(category_matches)
                    })
            
            # Determine primary category (most keyword matches)
            primary_category = "Uncategorized"
            if post_categories:
                primary_category = max(post_categories, key=lambda x: x['keyword_count'])['category']
                category_counts[primary_category] += 1
            else:
                category_counts["Uncategorized"] += 1
            
            result = {
                'ID': row['ID'],
                'Statement': row['Statement'],
                'Primary_Category': primary_category,
                'All_Categories': [cat['category'] for cat in post_categories],
                'Keywords_Found': found_keywords,
                'Keyword_Count': len(found_keywords)
            }
            
            # Add ground truth if available
            if self.ground_truth_available and 'True_Category' in row:
                result['True_Category'] = row['True_Category']
            
            results.append(result)
        
        self.classification_results = results
        self.keyword_frequency = dict(keyword_counts)
        
        # Calculate metrics if ground truth is available
        if self.ground_truth_available:
            self._calculate_metrics()
        
        return {
            'results': results,
            'category_counts': dict(category_counts),
            'keyword_frequency': dict(keyword_counts)
        }
    
    def _calculate_metrics(self):
        """Calculate precision, recall, F1 scores and other metrics"""
        if not self.classification_results or not self.ground_truth_available:
            return
        
        # Prepare data for metrics calculation
        y_true = []
        y_pred = []
        
        for result in self.classification_results:
            if pd.notna(result.get('True_Category')):
                y_true.append(result['True_Category'])
                y_pred.append(result['Primary_Category'])
        
        if not y_true:
            st.warning("No valid ground truth data found for metrics calculation")
            return
        
        # Get all unique categories
        all_categories = list(set(y_true + y_pred))
        
        # Calculate overall metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Accuracy
        accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
        
        # Per-category metrics
        classification_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=all_categories)
        
        self.metrics = {
            'overall': {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_micro': f1_micro
            },
            'per_category': classification_rep,
            'confusion_matrix': conf_matrix,
            'categories': all_categories,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def get_metrics_summary_df(self):
        """Get metrics summary as DataFrame"""
        if not self.metrics:
            return None
        
        per_cat = self.metrics['per_category']
        
        # Create per-category metrics DataFrame
        rows = []
        for category, metrics in per_cat.items():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                rows.append({
                    'Category': category,
                    'Precision': f"{metrics.get('precision', 0):.3f}",
                    'Recall': f"{metrics.get('recall', 0):.3f}",
                    'F1-Score': f"{metrics.get('f1-score', 0):.3f}",
                    'Support': int(metrics.get('support', 0))
                })
        
        return pd.DataFrame(rows)
    
    def get_confusion_matrix_fig(self):
        """Create confusion matrix visualization"""
        if not self.metrics or 'confusion_matrix' not in self.metrics:
            return None
        
        conf_matrix = self.metrics['confusion_matrix']
        categories = self.metrics['categories']
        
        # Create heatmap
        fig = px.imshow(
            conf_matrix,
            x=categories,
            y=categories,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(categories)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(conf_matrix[i][j]),
                    showarrow=False,
                    font=dict(color="white" if conf_matrix[i][j] > conf_matrix.max()/2 else "black")
                )
        
        fig.update_layout(
            xaxis_title="Predicted Category",
            yaxis_title="True Category",
            height=500
        )
        
        return fig
    
    def get_metrics_comparison_fig(self):
        """Create metrics comparison visualization"""
        if not self.metrics:
            return None
        
        per_cat = self.metrics['per_category']
        
        categories = []
        precision_vals = []
        recall_vals = []
        f1_vals = []
        
        for category, metrics in per_cat.items():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                categories.append(category)
                precision_vals.append(metrics.get('precision', 0))
                recall_vals.append(metrics.get('recall', 0))
                f1_vals.append(metrics.get('f1-score', 0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Precision', x=categories, y=precision_vals))
        fig.add_trace(go.Bar(name='Recall', x=categories, y=recall_vals))
        fig.add_trace(go.Bar(name='F1-Score', x=categories, y=f1_vals))
        
        fig.update_layout(
            barmode='group',
            title='Per-Category Metrics Comparison',
            xaxis_title='Category',
            yaxis_title='Score',
            height=500,
            xaxis={'categoryorder': 'total descending'}
        )
        
        return fig
    
    def get_category_analysis_df(self):
        """Get category frequency analysis as DataFrame"""
        if not self.classification_results:
            return None
            
        category_counts = defaultdict(int)
        for result in self.classification_results:
            category_counts[result['Primary_Category']] += 1
        
        total_posts = len(self.classification_results)
        
        category_df = pd.DataFrame([
            {
                'Category': category,
                'Posts': count,
                'Percentage': f"{(count/total_posts)*100:.1f}%"
            }
            for category, count in sorted(category_counts.items(), 
                                        key=lambda x: x[1], reverse=True)
        ])
        
        return category_df
    
    def get_keyword_frequency_df(self):
        """Get keyword frequency analysis as DataFrame"""
        if not self.keyword_frequency:
            return None
            
        keyword_df = pd.DataFrame([
            {
                'Keyword': keyword,
                'Frequency': count
            }
            for keyword, count in sorted(self.keyword_frequency.items(), 
                                       key=lambda x: x[1], reverse=True)
        ])
        
        return keyword_df
    
    def get_detailed_results_df(self):
        """Get detailed classification results as DataFrame"""
        if not self.classification_results:
            return None
            
        return pd.DataFrame(self.classification_results)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = CategoryClassifier()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

def data_upload_page():
    st.markdown("<h2 class='section-header'>📁 Data Upload</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload your Instagram data files to get started:
    1. **Posts Data**: CSV with columns 'ID' and 'Statement'
    2. **Ground Truth** (optional): CSV with columns 'ID' and 'True_Category' for metrics calculation
    3. **Categories** (optional): JSON file or manual configuration
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📄 Posts Data")
        posts_file = st.file_uploader("Upload posts CSV", type=['csv'], key="posts")
        
        if posts_file:
            try:
                posts_df = pd.read_csv(posts_file)
                st.success(f"✅ Loaded {len(posts_df)} posts")
                st.dataframe(posts_df.head())
                
                required_cols = ['ID', 'Statement']
                missing_cols = [col for col in required_cols if col not in posts_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.posts_df = posts_df
                    
            except Exception as e:
                st.error(f"Error loading posts file: {str(e)}")
    
    with col2:
        st.subheader("📊 Ground Truth (Optional)")
        ground_truth_file = st.file_uploader("Upload ground truth CSV", type=['csv'], key="ground_truth")
        
        if ground_truth_file:
            try:
                ground_truth_df = pd.read_csv(ground_truth_file)
                st.success(f"✅ Loaded ground truth for {len(ground_truth_df)} posts")
                st.dataframe(ground_truth_df.head())
                
                required_cols = ['ID', 'True_Category']
                missing_cols = [col for col in required_cols if col not in ground_truth_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.ground_truth_df = ground_truth_df
                    unique_categories = ground_truth_df['True_Category'].unique()
                    st.info(f"Found categories: {', '.join(unique_categories)}")
                    
            except Exception as e:
                st.error(f"Error loading ground truth file: {str(e)}")
    
    with col3:
        st.subheader("🏷️ Categories Configuration")
        
        category_option = st.radio(
            "Choose category source:",
            ["Use default categories", "Upload categories JSON", "Configure manually"]
        )
        
        categories_dict = None
        
        if category_option == "Upload categories JSON":
            categories_file = st.file_uploader("Upload categories JSON", type=['json'], key="categories")
            if categories_file:
                try:
                    categories_dict = json.load(categories_file)
                    st.success(f"✅ Loaded {len(categories_dict)} categories")
                    for cat, keywords in categories_dict.items():
                        st.write(f"**{cat}**: {len(keywords)} keywords")
                except Exception as e:
                    st.error(f"Error loading categories file: {str(e)}")
                    
        elif category_option == "Configure manually":
            st.write("Add custom categories:")
            
            if 'custom_categories' not in st.session_state:
                st.session_state.custom_categories = {}
            
            new_category = st.text_input("Category name:")
            new_keywords = st.text_area("Keywords (comma-separated):", 
                                      placeholder="keyword1, keyword2, keyword3")
            
            if st.button("Add Category"):
                if new_category and new_keywords:
                    keywords_list = [kw.strip() for kw in new_keywords.split(',') if kw.strip()]
                    st.session_state.custom_categories[new_category] = keywords_list
                    st.success(f"Added category: {new_category}")
            
            if st.session_state.custom_categories:
                st.write("**Current Categories:**")
                for cat, keywords in st.session_state.custom_categories.items():
                    st.write(f"• **{cat}**: {', '.join(keywords)}")
                categories_dict = st.session_state.custom_categories
    
    # Load data button
    if st.button("🚀 Load Data", type="primary"):
        if 'posts_df' in st.session_state:
            ground_truth_df = st.session_state.get('ground_truth_df', None)
            success = st.session_state.classifier.load_data_from_uploads(
                st.session_state.posts_df,
                categories_dict,
                ground_truth_df
            )
            if success:
                st.session_state.data_loaded = True
                st.success("🎉 Data loaded successfully! Go to Category Analysis.")
        else:
            st.error("❌ Please upload posts data file.")

def metrics_page():
    st.markdown("<h2 class='section-header'>📊 Performance Metrics</h2>", unsafe_allow_html=True)
    
    if not st.session_state.analysis_done:
        st.warning("⚠️ Please run category analysis first.")
        return
    
    if not st.session_state.classifier.ground_truth_available:
        st.warning("⚠️ Ground truth data is required for metrics calculation. Please upload ground truth data in the Data Upload section.")
        return
    
    if not st.session_state.classifier.metrics:
        st.error("❌ No metrics available. Please ensure ground truth data is properly formatted.")
        return
    
    # Overall Metrics
    st.markdown("<h3 class='section-header'>🎯 Overall Performance</h3>", unsafe_allow_html=True)
    
    overall_metrics = st.session_state.classifier.metrics['overall']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h2>{overall_metrics['accuracy']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Macro F1</h3>
            <h2>{overall_metrics['f1_macro']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Macro Precision</h3>
            <h2>{overall_metrics['precision_macro']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Macro Recall</h3>
            <h2>{overall_metrics['recall_macro']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Micro vs Macro comparison
    st.markdown("<h3 class='section-header'>📈 Micro vs Macro Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Macro Average** (treats all classes equally)")
        st.write(f"• Precision: {overall_metrics['precision_macro']:.3f}")
        st.write(f"• Recall: {overall_metrics['recall_macro']:.3f}")
        st.write(f"• F1-Score: {overall_metrics['f1_macro']:.3f}")
    
    with col2:
        st.markdown("**Micro Average** (weighted by class frequency)")
        st.write(f"• Precision: {overall_metrics['precision_micro']:.3f}")
        st.write(f"• Recall: {overall_metrics['recall_micro']:.3f}")
        st.write(f"• F1-Score: {overall_metrics['f1_micro']:.3f}")
    
    # Per-category metrics
    st.markdown("<h3 class='section-header'>🏷️ Per-Category Performance</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metrics Table")
        metrics_df = st.session_state.classifier.get_metrics_summary_df()
        if metrics_df is not None:
            st.dataframe(metrics_df, use_container_width=True)
            
            # Download metrics
            csv_data = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv_data,
                file_name="classification_metrics.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("Metrics Visualization")
        metrics_fig = st.session_state.classifier.get_metrics_comparison_fig()
        if metrics_fig:
            st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("<h3 class='section-header'>🔄 Confusion Matrix</h3>", unsafe_allow_html=True)
    
    conf_matrix_fig = st.session_state.classifier.get_confusion_matrix_fig()
    if conf_matrix_fig:
        st.plotly_chart(conf_matrix_fig, use_container_width=True)
    
    # Error Analysis
    st.markdown("<h3 class='section-header'>🔍 Error Analysis</h3>", unsafe_allow_html=True)
    
    detailed_df = st.session_state.classifier.get_detailed_results_df()
    if detailed_df is not None and 'True_Category' in detailed_df.columns:
        # Find misclassified examples
        errors_df = detailed_df[detailed_df['Primary_Category'] != detailed_df['True_Category']]
        
        st.write(f"**Total Misclassifications**: {len(errors_df)} out of {len(detailed_df)} ({len(errors_df)/len(detailed_df)*100:.1f}%)")
        
        if len(errors_df) > 0:
            # Show error breakdown
            error_summary = errors_df.groupby(['True_Category', 'Primary_Category']).size().reset_index(name='Count')
            error_summary = error_summary.sort_values('Count', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Most Common Misclassifications")
                st.dataframe(error_summary.head(10), use_container_width=True)
            
            with col2:
                st.subheader("Sample Misclassified Posts")
                sample_errors = errors_df.sample(min(5, len(errors_df)))
                for idx, row in sample_errors.iterrows():
                    st.write(f"**True**: {row['True_Category']} → **Predicted**: {row['Primary_Category']}")
                    st.write(f"*Statement*: {row['Statement'][:100]}...")
                    st.write("---")

def category_analysis_page():
    st.markdown("<h2 class='section-header'>🔍 Category Analysis</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Upload section.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        info_text = f"""
        **Current Data:**
        - Posts: {len(st.session_state.classifier.df)}
        - Categories: {len(st.session_state.classifier.categories)}
        """
        
        if st.session_state.classifier.ground_truth_available:
            info_text += "\n- Ground Truth: ✅ Available (metrics will be calculated)"
        else:
            info_text += "\n- Ground Truth: ❌ Not available"
        
        st.info(info_text)
    
    with col2:
        if st.button("🔍 Analyze Categories", type="primary"):
            with st.spinner("Analyzing post categories..."):
                results = st.session_state.classifier.classify_posts()
                
                if results:
                    st.session_state.analysis_done = True
                    st.success("✅ Analysis completed!")
                    if st.session_state.classifier.ground_truth_available:
                        st.success("📊 Performance metrics calculated! Check the Metrics tab.")
    
    if st.session_state.analysis_done:
        st.markdown("<h3 class='section-header'>📊 Category Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Frequency:")
            category_df = st.session_state.classifier.get_category_analysis_df()
            if category_df is not None:
                st.dataframe(category_df, use_container_width=True)
                
                # Download button for category analysis
                csv_data = category_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Category Analysis CSV",
                    data=csv_data,
                    file_name="category_analysis.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("Top Keywords Overall:")
            keyword_df = st.session_state.classifier.get_keyword_frequency_df()
            if keyword_df is not None:
                st.dataframe(keyword_df.head(10), use_container_width=True)
                
                # Download button for keyword frequency
                csv_data = keyword_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Keyword Frequency CSV",
                    data=csv_data,
                    file_name="keyword_frequency.csv",
                    mime="text/csv"
                )
        
        # Detailed results
        st.subheader("📝 Detailed Classification Results")
        
        detailed_df = st.session_state.classifier.get_detailed_results_df()
        if detailed_df is not None:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                categories_list = list(st.session_state.classifier.categories.keys()) + ["Uncategorized"]
                selected_category = st.selectbox(
                    "Filter by category:",
                    ["All"] + categories_list
                )
            
            with col2:
                min_keywords = st.slider("Min keywords found:", 0, 10, 0)
            
            with col3:
                show_top = st.slider("Show top N posts:", 10, 100, 20)
            
            # Apply filters
            filtered_df = detailed_df.copy()
            if selected_category != "All":
                filtered_df = filtered_df[filtered_df['Primary_Category'] == selected_category]
            
            filtered_df = filtered_df[filtered_df['Keyword_Count'] >= min_keywords]
            filtered_df = filtered_df.head(show_top)
            
            # Display results
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button for detailed results
            csv_data = detailed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Results CSV",
                data=csv_data,
                file_name="detailed_classification_results.csv",
                mime="text/csv"
            )

def main():
    st.markdown("<h1 class='main-header'>📊 Instagram Category Classifier</h1>", unsafe_allow_html=True)
    st.markdown("### Analyze Instagram posts by categorizing them based on keyword patterns")
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📁 Data Upload", "🔍 Category Analysis", "📊 Performance Metrics"]
    )
    
    # Show current categories in sidebar
    if st.session_state.data_loaded:
        st.sidebar.markdown("### 🏷️ Current Categories")
        for category, keywords in st.session_state.classifier.categories.items():
            with st.sidebar.expander(category):
                st.write(", ".join(keywords))
        
        # Show data status
        st.sidebar.markdown("### 📈 Data Status")
        st.sidebar.write(f"Posts loaded: {len(st.session_state.classifier.df)}")
        if st.session_state.classifier.ground_truth_available:
            st.sidebar.write("Ground truth: ✅")
        else:
            st.sidebar.write("Ground truth: ❌")
        
        if st.session_state.analysis_done:
            st.sidebar.write("Analysis: ✅ Complete")
            if st.session_state.classifier.metrics:
                accuracy = st.session_state.classifier.metrics['overall']['accuracy']
                st.sidebar.write(f"Accuracy: {accuracy:.3f}")
        else:
            st.sidebar.write("Analysis: ⏳ Pending")
    
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "🔍 Category Analysis":
        category_analysis_page()
    elif page == "📊 Performance Metrics":
        metrics_page()

if __name__ == "__main__":
    main()
