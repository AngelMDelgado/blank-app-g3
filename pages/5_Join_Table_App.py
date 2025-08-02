import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Instagram Join Table Analyzer",
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
</style>
""", unsafe_allow_html=True)

class InstagramJoinAnalyzer:
    def __init__(self):
        self.posts_df = None
        self.engagement_df = None
        self.joined_df = None
        self.keywords = []
        self.keyword_stats = {}

    def load_data_from_uploads(self, posts_file, engagement_file, custom_keywords=None):
        """Load data from Streamlit file uploads"""
        try:
            # Load posts data
            self.posts_df = pd.read_csv(posts_file)
            st.success(f"✅ Posts data loaded: {self.posts_df.shape[0]} rows, {self.posts_df.shape[1]} columns")
            
            required_cols = ['ID', 'Statement']
            missing_cols = [col for col in required_cols if col not in self.posts_df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns in posts data: {missing_cols}")
                return False
                
            # Load engagement data
            self.engagement_df = pd.read_csv(engagement_file)
            st.success(f"✅ Engagement data loaded: {self.engagement_df.shape[0]} rows, {self.engagement_df.shape[1]} columns")
            
            required_eng_cols = ['shortcode', 'number_likes', 'number_comments']
            missing_eng_cols = [col for col in required_eng_cols if col not in self.engagement_df.columns]
            if missing_eng_cols:
                st.error(f"❌ Missing required engagement columns: {missing_eng_cols}")
                return False
            
            # Setup keywords
            self.setup_keywords(custom_keywords)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False

    def join_datasets(self, join_type='inner'):
        """Join posts and engagement datasets"""
        if self.posts_df is None or self.engagement_df is None:
            st.error("❌ Both datasets must be loaded first")
            return False

        st.info(f"🔗 Joining datasets using {join_type} join...")
        
        # Perform the join
        self.joined_df = pd.merge(
            self.posts_df,
            self.engagement_df,
            left_on='ID',
            right_on='shortcode',
            how=join_type
        )

        # Display join statistics
        posts_matched = self.joined_df.shape[0]
        posts_total = self.posts_df.shape[0]
        engagement_total = self.engagement_df.shape[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Posts Matched", f"{posts_matched}/{posts_total}")
        with col2:
            st.metric("Match Rate", f"{posts_matched/posts_total*100:.1f}%")
        with col3:
            st.metric("Final Dataset Size", f"{self.joined_df.shape[0]} rows")

        return True

    def explore_joined_data(self):
        """Explore the joined dataset"""
        if self.joined_df is None:
            st.error("❌ No joined data available")
            return

        st.subheader("🔍 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", self.joined_df.shape[0])
        with col2:
            st.metric("Total Columns", self.joined_df.shape[1])
        with col3:
            memory_mb = self.joined_df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")

        # Data types
        st.subheader("📋 Data Types")
        dtype_df = pd.DataFrame({
            'Column': self.joined_df.dtypes.index,
            'Data Type': self.joined_df.dtypes.values
        })
        st.dataframe(dtype_df, use_container_width=True)

        # Missing values
        st.subheader("❓ Missing Values Analysis")
        missing = self.joined_df.isnull().sum()
        missing_pct = (missing / len(self.joined_df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

        # Display sample data
        st.subheader("👀 Sample Data")
        display_cols = ['ID', 'Statement', 'number_likes', 'number_comments']
        available_cols = [col for col in display_cols if col in self.joined_df.columns]
        st.dataframe(self.joined_df[available_cols].head(10), use_container_width=True)

        # Engagement statistics
        if 'number_likes' in self.joined_df.columns and 'number_comments' in self.joined_df.columns:
            # Ensure numeric columns
            self.joined_df['number_likes'] = pd.to_numeric(self.joined_df['number_likes'], errors='coerce').fillna(0)
            self.joined_df['number_comments'] = pd.to_numeric(self.joined_df['number_comments'], errors='coerce').fillna(0)
            self.joined_df['total_engagement'] = self.joined_df['number_likes'] + self.joined_df['number_comments']

            st.subheader("📈 Engagement Statistics")
            engagement_stats = self.joined_df[['number_likes', 'number_comments', 'total_engagement']].describe()
            st.dataframe(engagement_stats, use_container_width=True)

    def visualize_engagement(self):
        """Create visualizations for engagement data using Streamlit charts"""
        if self.joined_df is None or 'total_engagement' not in self.joined_df.columns:
            st.error("❌ No engagement data available for visualization")
            return

        st.subheader("📊 Engagement Visualizations")

        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement distribution histogram
            st.subheader("Total Engagement Distribution")
            st.bar_chart(pd.Series(self.joined_df['total_engagement']).value_counts().sort_index())
            
            # Likes vs Comments scatter
            st.subheader("Likes vs Comments")
            scatter_data = pd.DataFrame({
                'Likes': self.joined_df['number_likes'],
                'Comments': self.joined_df['number_comments']
            })
            st.scatter_chart(scatter_data, x='Likes', y='Comments')

        with col2:
            # Box plot alternative - show statistics
            st.subheader("Engagement Statistics Summary")
            likes_stats = self.joined_df['number_likes'].describe()
            comments_stats = self.joined_df['number_comments'].describe()
            
            stats_df = pd.DataFrame({
                'Likes': likes_stats,
                'Comments': comments_stats
            })
            st.dataframe(stats_df)
            
            # Top posts by engagement
            st.subheader("Top Posts by Engagement")
            top_posts = self.joined_df.nlargest(min(10, len(self.joined_df)), 'total_engagement')
            top_chart_data = pd.DataFrame({
                'Post': [f"Post {i+1}" for i in range(len(top_posts))],
                'Total Engagement': top_posts['total_engagement'].values
            })
            st.bar_chart(top_chart_data.set_index('Post'))

        # Display top performing posts details
        st.subheader(f"🏆 Top {len(top_posts)} Performing Posts")
        
        for idx, (_, row) in enumerate(top_posts.head().iterrows(), 1):
            with st.expander(f"Post {idx} - Engagement: {row['total_engagement']:.0f}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    statement_preview = str(row['Statement'])[:200] + "..." if len(str(row['Statement'])) > 200 else str(row['Statement'])
                    st.write(f"**Statement:** {statement_preview}")
                with col2:
                    st.metric("Likes", f"{row['number_likes']:.0f}")
                    st.metric("Comments", f"{row['number_comments']:.0f}")

    def setup_keywords(self, custom_keywords=None):
        """Setup keywords for analysis"""
        if custom_keywords:
            self.keywords = [kw.strip().lower() for kw in custom_keywords if kw.strip()]
        else:
            # Default personalization keywords
            self.keywords = [
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
        
        return self.keywords

    def analyze_keywords(self, min_word_count=5):
        """Analyze keyword performance"""
        if self.joined_df is None:
            st.error("❌ No joined data available")
            return

        if not self.keywords:
            self.setup_keywords()

        keyword_engagement = {}
        keyword_frequency = {}
        post_features = []

        progress_bar = st.progress(0)
        total_rows = len(self.joined_df)

        for idx, row in self.joined_df.iterrows():
            progress_bar.progress((idx + 1) / total_rows)
            
            if pd.isna(row['Statement']):
                continue

            text = str(row['Statement']).lower()
            words = re.findall(r'\b\w+\b', text)

            if len(words) < min_word_count:
                continue

            # Count keywords in this post
            post_keywords = []
            for keyword in self.keywords:
                if keyword in words:
                    post_keywords.append(keyword)
                    # Track frequency
                    if keyword not in keyword_frequency:
                        keyword_frequency[keyword] = 0
                    keyword_frequency[keyword] += 1

                    # Track engagement for this keyword
                    if keyword not in keyword_engagement:
                        keyword_engagement[keyword] = []

                    engagement = row['number_likes'] + row['number_comments']
                    keyword_engagement[keyword].append(engagement)

            # Create post features
            post_feature = {
                'post_id': row['ID'],
                'statement_preview': str(row['Statement'])[:100] + "..." if len(str(row['Statement'])) > 100 else str(row['Statement']),
                'word_count': len(words),
                'keyword_count': len(post_keywords),
                'keywords_found': post_keywords,
                'keyword_density': len(post_keywords) / len(words) if words else 0,
                'likes': row['number_likes'],
                'comments': row['number_comments'],
                'total_engagement': row['number_likes'] + row['number_comments']
            }

            post_features.append(post_feature)

        progress_bar.empty()

        # Calculate average engagement per keyword
        keyword_avg_engagement = {}
        for keyword, engagements in keyword_engagement.items():
            if engagements:
                keyword_avg_engagement[keyword] = np.mean(engagements)

        self.keyword_stats = {
            'frequency': keyword_frequency,
            'avg_engagement': keyword_avg_engagement,
            'post_features': post_features
        }

        return self.keyword_stats

    def display_keyword_results(self):
        """Display keyword analysis results"""
        if not self.keyword_stats or not self.keyword_stats['avg_engagement']:
            st.warning("❌ No keyword analysis data available")
            return

        st.subheader("📊 Keyword Performance Results")

        # Top performing keywords
        sorted_keywords = sorted(self.keyword_stats['avg_engagement'].items(), key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("🏆 **Top 10 Keywords by Average Engagement:**")
            top_perf_data = []
            for i, (keyword, avg_eng) in enumerate(sorted_keywords[:10], 1):
                freq = self.keyword_stats['frequency'].get(keyword, 0)
                top_perf_data.append({
                    'Rank': i,
                    'Keyword': keyword,
                    'Avg Engagement': f"{avg_eng:.1f}",
                    'Frequency': freq
                })
            st.dataframe(pd.DataFrame(top_perf_data), use_container_width=True)

        with col2:
            st.write("📈 **Top 10 Most Frequent Keywords:**")
            sorted_freq = sorted(self.keyword_stats['frequency'].items(), key=lambda x: x[1], reverse=True)
            top_freq_data = []
            for i, (keyword, freq) in enumerate(sorted_freq[:10], 1):
                avg_eng = self.keyword_stats['avg_engagement'].get(keyword, 0)
                top_freq_data.append({
                    'Rank': i,
                    'Keyword': keyword,
                    'Frequency': freq,
                    'Avg Engagement': f"{avg_eng:.1f}"
                })
            st.dataframe(pd.DataFrame(top_freq_data), use_container_width=True)

    def visualize_keyword_performance(self):
        """Visualize keyword performance using Streamlit charts"""
        if not self.keyword_stats or not self.keyword_stats['avg_engagement']:
            st.error("❌ No keyword analysis data available")
            return

        st.subheader("📈 Keyword Performance Visualizations")

        # Prepare data for visualization
        keywords = list(self.keyword_stats['avg_engagement'].keys())
        avg_engagements = list(self.keyword_stats['avg_engagement'].values())
        frequencies = [self.keyword_stats['frequency'].get(kw, 0) for kw in keywords]

        col1, col2 = st.columns(2)

        with col1:
            # Top keywords by engagement
            st.subheader("Top 15 Keywords by Average Engagement")
            top_keywords_data = sorted(zip(keywords, avg_engagements), key=lambda x: x[1], reverse=True)[:15]
            if top_keywords_data:
                top_kw, top_eng = zip(*top_keywords_data)
                chart_data = pd.DataFrame({
                    'Keyword': top_kw,
                    'Average Engagement': top_eng
                })
                st.bar_chart(chart_data.set_index('Keyword'))

            # Engagement vs Frequency scatter plot
            st.subheader("Keyword Frequency vs Average Engagement")
            scatter_data = pd.DataFrame({
                'Frequency': frequencies,
                'Average Engagement': avg_engagements
            })
            st.scatter_chart(scatter_data, x='Frequency', y='Average Engagement')

        with col2:
            # Top frequent keywords
            st.subheader("Top 15 Most Frequent Keywords")
            top_freq_data = sorted(zip(keywords, frequencies), key=lambda x: x[1], reverse=True)[:15]
            if top_freq_data:
                top_freq_kw, top_freq_counts = zip(*top_freq_data)
                freq_chart_data = pd.DataFrame({
                    'Keyword': top_freq_kw,
                    'Frequency': top_freq_counts
                })
                st.bar_chart(freq_chart_data.set_index('Keyword'))

            # Keyword frequency distribution
            st.subheader("Keyword Frequency Distribution")
            freq_dist = pd.Series(frequencies).value_counts().sort_index()
            freq_dist_df = pd.DataFrame({
                'Frequency Range': freq_dist.index,
                'Number of Keywords': freq_dist.values
            })
            st.bar_chart(freq_dist_df.set_index('Frequency Range'))

    def predict_engagement(self, text):
        """Predict engagement for new text"""
        if not self.keyword_stats or not self.keyword_stats['avg_engagement']:
            return "No analysis data available", "Please run keyword analysis first"

        words = re.findall(r'\b\w+\b', text.lower())
        found_keywords = [kw for kw in self.keywords if kw in words]

        if not found_keywords:
            return "Low", "No target keywords found"

        # Calculate predicted engagement based on keyword averages
        keyword_scores = []
        for kw in found_keywords:
            if kw in self.keyword_stats['avg_engagement']:
                keyword_scores.append(self.keyword_stats['avg_engagement'][kw])

        if not keyword_scores:
            return "Medium", "Keywords found but no historical data"

        avg_predicted_engagement = np.mean(keyword_scores)

        # Get percentiles for classification
        if self.joined_df is not None and 'total_engagement' in self.joined_df.columns:
            percentiles = np.percentile(self.joined_df['total_engagement'], [25, 50, 75, 90])

            if avg_predicted_engagement >= percentiles[2]:  # 75th percentile
                return "High", f"Predicted engagement: {avg_predicted_engagement:.0f} (Keywords: {', '.join(found_keywords)})"
            elif avg_predicted_engagement >= percentiles[1]:  # 50th percentile
                return "Medium", f"Predicted engagement: {avg_predicted_engagement:.0f} (Keywords: {', '.join(found_keywords)})"
            else:
                return "Low", f"Predicted engagement: {avg_predicted_engagement:.0f} (Keywords: {', '.join(found_keywords)})"

        return "Medium", f"Predicted engagement: {avg_predicted_engagement:.0f} (Keywords: {', '.join(found_keywords)})"

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = InstagramJoinAnalyzer()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_joined' not in st.session_state:
    st.session_state.data_joined = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

def data_upload_page():
    st.markdown("<h2 class='section-header'>📁 Data Upload</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload your Instagram data files to get started:
    1. **Posts Data**: CSV with columns 'ID' and 'Statement'
    2. **Engagement Data**: CSV with columns 'shortcode', 'number_likes', 'number_comments'
    3. **Keywords** (optional): Custom keywords for analysis
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Posts Data")
        posts_file = st.file_uploader("Upload posts CSV", type=['csv'], key="posts")
        
        if posts_file:
            try:
                posts_df = pd.read_csv(posts_file)
                st.success(f"✅ Loaded {len(posts_df)} posts")
                st.dataframe(posts_df.head(), use_container_width=True)
                
                required_cols = ['ID', 'Statement']
                missing_cols = [col for col in required_cols if col not in posts_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.posts_file = posts_file
                    
            except Exception as e:
                st.error(f"Error loading posts file: {str(e)}")

    with col2:
        st.subheader("📈 Engagement Data")
        engagement_file = st.file_uploader("Upload engagement CSV", type=['csv'], key="engagement")
        
        if engagement_file:
            try:
                engagement_df = pd.read_csv(engagement_file)
                st.success(f"✅ Loaded {len(engagement_df)} engagement records")
                st.dataframe(engagement_df.head(), use_container_width=True)
                
                required_cols = ['shortcode', 'number_likes', 'number_comments']
                missing_cols = [col for col in required_cols if col not in engagement_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.engagement_file = engagement_file
                    
            except Exception as e:
                st.error(f"Error loading engagement file: {str(e)}")

    # Keywords section
    st.subheader("🔑 Keywords Configuration")
    
    keyword_option = st.radio(
        "Choose keyword source:",
        ["Use default keywords", "Enter custom keywords"]
    )
    
    custom_keywords = None
    
    if keyword_option == "Enter custom keywords":
        keywords_text = st.text_area(
            "Enter keywords (comma-separated):",
            height=150,
            placeholder="personalized, custom, tailored, bespoke, individual, unique..."
        )
        if keywords_text:
            custom_keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            st.info(f"📝 {len(custom_keywords)} custom keywords entered")
    else:
        st.info("📝 Using default personalization keywords")

    # Load data button
    if st.button("🚀 Load Data", type="primary"):
        if 'posts_file' in st.session_state and 'engagement_file' in st.session_state:
            success = st.session_state.analyzer.load_data_from_uploads(
                st.session_state.posts_file,
                st.session_state.engagement_file,
                custom_keywords
            )
            if success:
                st.session_state.data_loaded = True
                st.success("🎉 Data loaded successfully! Go to 'Join & Explore' section.")
        else:
            st.error("❌ Please upload both posts and engagement data files.")

def join_explore_page():
    st.markdown("<h2 class='section-header'>🔗 Join & Explore Data</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Upload section.")
        return

    # Join type selection
    st.subheader("🔗 Dataset Joining")
    join_type = st.selectbox(
        "Select join type:",
        ["inner", "left", "right", "outer"],
        help="Inner: Only matching records | Left: All posts + matching engagement | Right: All engagement + matching posts | Outer: All records"
    )

    if st.button("🔗 Join Datasets", type="primary"):
        if st.session_state.analyzer.join_datasets(join_type):
            st.session_state.data_joined = True
            st.success("✅ Datasets joined successfully!")

    if st.session_state.data_joined:
        # Explore joined data
        st.markdown("---")
        st.session_state.analyzer.explore_joined_data()
        
        # Visualize engagement
        st.markdown("---")
        st.session_state.analyzer.visualize_engagement()

def keyword_analysis_page():
    st.markdown("<h2 class='section-header'>📊 Keyword Analysis</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_joined:
        st.warning("⚠️ Please join the datasets first in the 'Join & Explore' section.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        min_word_count = st.slider("Minimum word count per post", 1, 20, 5)
    
    with col2:
        st.info(f"""
        **Current Setup:**
        - Keywords: {len(st.session_state.analyzer.keywords)}
        - Joined Dataset: {st.session_state.analyzer.joined_df.shape[0] if st.session_state.analyzer.joined_df is not None else 0} rows
        """)

    if st.button("🔍 Analyze Keywords", type="primary"):
        with st.spinner("Analyzing keyword performance..."):
            stats = st.session_state.analyzer.analyze_keywords(min_word_count)
            
            if stats:
                st.session_state.analysis_done = True
                st.success("✅ Analysis completed!")
                
                # Display results
                st.session_state.analyzer.display_keyword_results()
                
                # Show visualizations
                st.markdown("---")
                st.session_state.analyzer.visualize_keyword_performance()

def prediction_page():
    st.markdown("<h2 class='section-header'>🔮 Engagement Prediction</h2>", unsafe_allow_html=True)
    
    if not st.session_state.analysis_done:
        st.warning("⚠️ Please complete keyword analysis first.")
        return

    st.subheader("📝 Test Your Instagram Post")
    
    sample_text = st.text_area(
        "Enter Instagram post text:",
        height=150,
        placeholder="Enter your Instagram post text here to predict its engagement potential..."
    )

    if sample_text and st.button("🔍 Predict Engagement", type="primary"):
        prediction, details = st.session_state.analyzer.predict_engagement(sample_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == "High":
                st.success(f"**Prediction: {prediction} Engagement** 🚀")
            elif prediction == "Medium":
                st.info(f"**Prediction: {prediction} Engagement** 📊")
            else:
                st.warning(f"**Prediction: {prediction} Engagement** 📉")
        
        with col2:
            st.info(f"**Details:** {details}")
        
        # Text analysis
        st.subheader("🔍 Text Analysis")
        
        text = sample_text.lower()
        words = re.findall(r'\b\w+\b', text)
        found_keywords = [kw for kw in st.session_state.analyzer.keywords if kw in words]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Word Count", len(words))
        
        with col2:
            st.metric("Keywords Found", len(found_keywords))
        
        with col3:
            density = len(found_keywords) / len(words) if words else 0
            st.metric("Keyword Density", f"{density:.3f}")
        
        with col4:
            features = sum([
                '!' in sample_text,
                '?' in sample_text,
                '#' in sample_text,
                '@' in sample_text
            ])
            st.metric("Special Features", features)
        
        if found_keywords:
            st.subheader("✅ Keywords Found")
            # Display keywords as tags
            keyword_tags = " ".join([f"`{kw}`" for kw in found_keywords])
            st.markdown(keyword_tags)
            
            # Show performance of found keywords
            if st.session_state.analyzer.keyword_stats['avg_engagement']:
                st.subheader("📊 Keyword Performance")
                perf_data = []
                for kw in found_keywords:
                    if kw in st.session_state.analyzer.keyword_stats['avg_engagement']:
                        perf_data.append({
                            'Keyword': kw,
                            'Avg Engagement': st.session_state.analyzer.keyword_stats['avg_engagement'][kw],
                            'Frequency': st.session_state.analyzer.keyword_stats['frequency'].get(kw, 0)
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(perf_df, use_container_width=True)

def export_page():
    st.markdown("<h2 class='section-header'>💾 Export Results</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_joined:
        st.warning("⚠️ No data available to export. Please complete the analysis first.")
        return

    st.subheader("📄 Available Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Joined Dataset")
        if st.session_state.analyzer.joined_df is not None:
            st.success(f"✅ {st.session_state.analyzer.joined_df.shape[0]} rows available")
            
            # Convert DataFrame to CSV
            csv_data = st.session_state.analyzer.joined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Joined Dataset (CSV)",
                data=csv_data,
                file_name="instagram_joined_data.csv",
                mime="text/csv"
            )
            
            # Preview
            with st.expander("👀 Preview Data"):
                st.dataframe(st.session_state.analyzer.joined_df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("🔑 Keyword Analysis")
        if st.session_state.analysis_done and st.session_state.analyzer.keyword_stats['avg_engagement']:
            keyword_results = pd.DataFrame([
                {
                    'keyword': kw,
                    'avg_engagement': avg_eng,
                    'frequency': st.session_state.analyzer.keyword_stats['frequency'].get(kw, 0)
                }
                for kw, avg_eng in st.session_state.analyzer.keyword_stats['avg_engagement'].items()
            ])
            
            st.success(f"✅ {len(keyword_results)} keywords analyzed")
            
            # Convert DataFrame to CSV
            keyword_csv = keyword_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Keyword Analysis (CSV)",
                data=keyword_csv,
                file_name="keyword_analysis_results.csv",
                mime="text/csv"
            )
            
            # Preview
            with st.expander("👀 Preview Keywords"):
                st.dataframe(keyword_results.head(10), use_container_width=True)
        else:
            st.info("⏳ Complete keyword analysis to enable export")

    # Summary Report
    st.markdown("---")
    st.subheader("📋 Analysis Summary Report")
    
    if st.button("📄 Generate Summary Report"):
        summary_report = generate_summary_report()
        
        st.download_button(
            label="📥 Download Summary Report (TXT)",
            data=summary_report,
            file_name="instagram_analysis_summary.txt",
            mime="text/plain"
        )
        
        with st.expander("👀 Preview Summary Report"):
            st.text(summary_report)

def generate_summary_report():
    """Generate a text summary report of the analysis"""
    analyzer = st.session_state.analyzer
    
    report = []
    report.append("INSTAGRAM JOIN TABLE ANALYZER - SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Dataset information
    report.append("DATASET INFORMATION")
    report.append("-" * 30)
    if analyzer.joined_df is not None:
        report.append(f"Total posts analyzed: {analyzer.joined_df.shape[0]}")
        report.append(f"Total columns: {analyzer.joined_df.shape[1]}")
        
        if 'total_engagement' in analyzer.joined_df.columns:
            total_eng = analyzer.joined_df['total_engagement'].sum()
            avg_eng = analyzer.joined_df['total_engagement'].mean()
            median_eng = analyzer.joined_df['total_engagement'].median()
            report.append(f"Total engagement: {total_eng:,.0f}")
            report.append(f"Average engagement per post: {avg_eng:.1f}")
            report.append(f"Median engagement per post: {median_eng:.1f}")
    
    report.append("")
    
    # Keyword analysis
    if analyzer.keyword_stats and analyzer.keyword_stats['avg_engagement']:
        report.append("KEYWORD ANALYSIS")
        report.append("-" * 30)
        report.append(f"Total keywords analyzed: {len(analyzer.keywords)}")
        report.append(f"Keywords found in posts: {len(analyzer.keyword_stats['avg_engagement'])}")
        
        # Top performing keywords
        report.append("\nTOP 10 KEYWORDS BY AVERAGE ENGAGEMENT:")
        sorted_keywords = sorted(analyzer.keyword_stats['avg_engagement'].items(), 
                               key=lambda x: x[1], reverse=True)
        for i, (keyword, avg_eng) in enumerate(sorted_keywords[:10], 1):
            freq = analyzer.keyword_stats['frequency'].get(keyword, 0)
            report.append(f"{i:2d}. {keyword:<20} | Avg: {avg_eng:6.1f} | Freq: {freq:3d}")
        
        # Most frequent keywords
        report.append("\nTOP 10 MOST FREQUENT KEYWORDS:")
        sorted_freq = sorted(analyzer.keyword_stats['frequency'].items(), 
                           key=lambda x: x[1], reverse=True)
        for i, (keyword, freq) in enumerate(sorted_freq[:10], 1):
            avg_eng = analyzer.keyword_stats['avg_engagement'].get(keyword, 0)
            report.append(f"{i:2d}. {keyword:<20} | Freq: {freq:3d} | Avg: {avg_eng:6.1f}")
    
    report.append("")
    report.append("END OF REPORT")
    report.append("=" * 60)
    
    return "\n".join(report)

def show_analysis_progress():
    """Show analysis progress and status"""
    progress_items = [
        ("📁 Data Upload", st.session_state.data_loaded),
        ("🔗 Join Datasets", st.session_state.data_joined),
        ("📊 Keyword Analysis", st.session_state.analysis_done),
    ]
    
    st.sidebar.markdown("### 📋 Analysis Progress")
    
    for item, completed in progress_items:
        if completed:
            st.sidebar.success(f"✅ {item}")
        else:
            st.sidebar.info(f"⏳ {item}")

def show_data_stats():
    """Show current data statistics in sidebar"""
    if st.session_state.data_loaded:
        analyzer = st.session_state.analyzer
        
        st.sidebar.markdown("### 📊 Data Statistics")
        
        if analyzer.posts_df is not None:
            st.sidebar.metric("Posts Loaded", len(analyzer.posts_df))
        
        if analyzer.engagement_df is not None:
            st.sidebar.metric("Engagement Records", len(analyzer.engagement_df))
        
        if analyzer.joined_df is not None:
            st.sidebar.metric("Joined Records", len(analyzer.joined_df))
        
        if analyzer.keywords:
            st.sidebar.metric("Keywords", len(analyzer.keywords))

def main():
    st.markdown("<h1 class='main-header'>📊 Instagram Join Table Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("### Analyze Instagram engagement based on personalized language patterns")

    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    
    # Show progress and stats
    show_analysis_progress()
    show_data_stats()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📁 Data Upload", "🔗 Join & Explore", "📊 Keyword Analysis", "🔮 Predictions", "💾 Export Results"]
    )

    # Add help section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ❓ Need Help?")
    
    with st.sidebar.expander("📋 Data Format Requirements"):
        st.markdown("""
        **Posts CSV must have:**
        - `ID`: Unique post identifier
        - `Statement`: Post text content
        
        **Engagement CSV must have:**
        - `shortcode`: Post identifier (matches ID)
        - `number_likes`: Number of likes
        - `number_comments`: Number of comments
        """)
    
    with st.sidebar.expander("🔧 How to Use"):
        st.markdown("""
        1. **Upload Data**: Upload your posts and engagement CSV files
        2. **Join & Explore**: Join datasets and explore the data
        3. **Keyword Analysis**: Analyze keyword performance
        4. **Predictions**: Test engagement predictions
        5. **Export**: Download your results
        """)
    
    with st.sidebar.expander("📊 Chart Information"):
        st.markdown("""
        **Chart Types Used:**
        - Bar charts for rankings and distributions
        - Scatter plots for correlations
        - Data tables for detailed statistics
        - Metrics for key performance indicators
        """)

    # Main page routing
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "🔗 Join & Explore":
        join_explore_page()
    elif page == "📊 Keyword Analysis":
        keyword_analysis_page()
    elif page == "🔮 Predictions":
        prediction_page()
    elif page == "💾 Export Results":
        export_page()

# Run the app
if __name__ == "__main__":
    main()
