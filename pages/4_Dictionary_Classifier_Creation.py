import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from collections import Counter, defaultdict

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
</style>
""", unsafe_allow_html=True)

class BasicClassifier:
    def __init__(self):
        self.df = None
        self.engagement_df = None
        self.keywords = []
        self.feature_matrix = None
        self.engagement_scores = {}
        self.keyword_stats = {}
        
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
                st.warning("⚠️ No engagement data provided - running in analysis mode only")
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
    
    def analyze_keywords(self, min_word_count=5):
        """Analyze keyword usage and engagement correlation"""
        keyword_engagement = {}
        keyword_frequency = {}
        post_features = []
        
        for idx, row in self.df.iterrows():
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
                    
                    if 'number_likes' in row and 'number_comments' in row:
                        engagement = row['number_likes'] + row['number_comments']
                        keyword_engagement[keyword].append(engagement)
            
            # Create post features
            post_feature = {
                'post_id': row['ID'],
                'statement': row['Statement'][:100] + "..." if len(row['Statement']) > 100 else row['Statement'],
                'word_count': len(words),
                'keyword_count': len(post_keywords),
                'keywords_found': post_keywords,
                'keyword_density': len(post_keywords) / len(words) if words else 0,
                'has_exclamation': '!' in row['Statement'],
                'has_question': '?' in row['Statement'],
                'has_hashtag': '#' in row['Statement'],
                'has_mention': '@' in row['Statement'],
            }
            
            if 'number_likes' in row and 'number_comments' in row:
                post_feature['likes'] = row['number_likes']
                post_feature['comments'] = row['number_comments']
                post_feature['total_engagement'] = row['number_likes'] + row['number_comments']
            
            post_features.append(post_feature)
        
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
    
    def get_engagement_percentiles(self):
        """Calculate engagement percentiles"""
        if 'number_likes' not in self.df.columns or 'number_comments' not in self.df.columns:
            return None
            
        engagements = self.df['number_likes'] + self.df['number_comments']
        
        percentiles = {
            '25th': np.percentile(engagements, 25),
            '50th': np.percentile(engagements, 50),
            '75th': np.percentile(engagements, 75),
            '90th': np.percentile(engagements, 90),
            '95th': np.percentile(engagements, 95)
        }
        
        return percentiles
    
    def predict_engagement_simple(self, text):
        """Simple engagement prediction based on keyword analysis"""
        if not self.keyword_stats:
            return None, "No analysis available"
            
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
        percentiles = self.get_engagement_percentiles()
        
        if percentiles:
            if avg_predicted_engagement >= percentiles['75th']:
                return "High", f"Predicted engagement: {avg_predicted_engagement:.0f}"
            elif avg_predicted_engagement >= percentiles['50th']:
                return "Medium", f"Predicted engagement: {avg_predicted_engagement:.0f}"
            else:
                return "Low", f"Predicted engagement: {avg_predicted_engagement:.0f}"
        
        return "Medium", f"Predicted engagement: {avg_predicted_engagement:.0f}"

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = BasicClassifier()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

def data_upload_page():
    st.markdown("<h2 class='section-header'>📁 Data Upload</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload your Instagram data files to get started:
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
        keywords_file = st.file_uploader("Upload keywords file", type=['txt'], key="keywords")
        if keywords_file:
            try:
                keywords_list = keywords_file.read().decode('utf-8').split('\n')
                keywords_list = [kw.strip() for kw in keywords_list if kw.strip()]
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
                st.success("🎉 Data loaded successfully! Go to Keyword Analysis.")
        else:
            st.error("❌ Please upload both posts and engagement data files.")

def keyword_analysis_page():
    st.markdown("<h2 class='section-header'>🔍 Keyword Analysis</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Upload section.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_word_count = st.slider("Minimum word count per post", 1, 20, 5)
    
    with col2:
        st.info(f"""
        **Current Data:**
        - Posts: {len(st.session_state.classifier.df)}
        - Keywords: {len(st.session_state.classifier.keywords)}
        """)
    
    if st.button("🔍 Analyze Keywords", type="primary"):
        with st.spinner("Analyzing keyword performance..."):
            stats = st.session_state.classifier.analyze_keywords(min_word_count)
            
            if stats:
                st.session_state.analysis_done = True
                st.success("✅ Analysis completed!")
                
                # Keyword frequency analysis
                st.subheader("📊 Keyword Frequency")
                
                if stats['frequency']:
                    freq_df = pd.DataFrame(list(stats['frequency'].items()), 
                                         columns=['Keyword', 'Frequency'])
                    freq_df = freq_df.sort_values('Frequency', ascending=False)
                    st.dataframe(freq_df.head(20))
                
                # Keyword engagement analysis
                if stats['avg_engagement']:
                    st.subheader("📈 Keyword Performance (Average Engagement)")
                    
                    eng_df = pd.DataFrame(list(stats['avg_engagement'].items()), 
                                        columns=['Keyword', 'Avg_Engagement'])
                    eng_df = eng_df.sort_values('Avg_Engagement', ascending=False)
                    st.dataframe(eng_df.head(20))
                    
                    # Best performing keywords
                    st.subheader("🏆 Top Performing Keywords")
                    top_keywords = eng_df.head(10)
                    
                    for idx, row in top_keywords.iterrows():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{row['Keyword']}**")
                        with col2:
                            st.metric("Avg Engagement", f"{row['Avg_Engagement']:.0f}")
                
                # Post analysis
                st.subheader("📝 Post Analysis")
                
                posts_df = pd.DataFrame(stats['post_features'])
                if not posts_df.empty:
                    # Filter options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        min_keywords = st.slider("Min keywords per post", 0, 10, 1)
                    with col2:
                        sort_by = st.selectbox("Sort by", 
                                             ['total_engagement', 'keyword_count', 'keyword_density'])
                    with col3:
                        show_top = st.slider("Show top N posts", 5, 50, 20)
                    
                    # Filter and sort
                    filtered_posts = posts_df[posts_df['keyword_count'] >= min_keywords]
                    if sort_by in filtered_posts.columns:
                        filtered_posts = filtered_posts.sort_values(sort_by, ascending=False)
                    
                    st.dataframe(filtered_posts.head(show_top))

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
        prediction, details = st.session_state.classifier.predict_engagement_simple(sample_text)
        
        if prediction:
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
            found_keywords = [kw for kw in st.session_state.classifier.keywords if kw in words]
            
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
                keyword_chips = " ".join([f"`{kw}`" for kw in found_keywords])
                st.markdown(keyword_chips)
                
                # Show performance of found keywords
                if st.session_state.classifier.keyword_stats['avg_engagement']:
                    st.subheader("📊 Keyword Performance")
                    perf_data = []
                    for kw in found_keywords:
                        if kw in st.session_state.classifier.keyword_stats['avg_engagement']:
                            perf_data.append({
                                'Keyword': kw,
                                'Avg_Engagement': st.session_state.classifier.keyword_stats['avg_engagement'][kw],
                                'Frequency': st.session_state.classifier.keyword_stats['frequency'].get(kw, 0)
                            })
                    
                    if perf_data:
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df)

def insights_page():
    st.markdown("<h2 class='section-header'>💡 Insights & Recommendations</h2>", unsafe_allow_html=True)
    
    if not st.session_state.analysis_done:
        st.warning("⚠️ Please complete keyword analysis first.")
        return
    
    stats = st.session_state.classifier.keyword_stats
    
    if stats['avg_engagement']:
        st.subheader("🎯 Key Insights")
        
        # Top insights
        eng_df = pd.DataFrame(list(stats['avg_engagement'].items()), 
                            columns=['Keyword', 'Avg_Engagement'])
        eng_df = eng_df.sort_values('Avg_Engagement', ascending=False)
        
        freq_df = pd.DataFrame(list(stats['frequency'].items()), 
                             columns=['Keyword', 'Frequency'])
        freq_df = freq_df.sort_values('Frequency', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**🏆 Highest Performing Keywords**")
            top_performers = eng_df.head(5)
            for _, row in top_performers.iterrows():
                st.write(f"• **{row['Keyword']}**: {row['Avg_Engagement']:.0f} avg engagement")
        
        with col2:
            st.info("**📈 Most Frequently Used**")
            most_used = freq_df.head(5)
            for _, row in most_used.iterrows():
                st.write(f"• **{row['Keyword']}**: Used {row['Frequency']} times")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        # Find underused high-performers
        merged_df = pd.merge(eng_df, freq_df, on='Keyword', how='inner')
        merged_df['performance_ratio'] = merged_df['Avg_Engagement'] / merged_df['Frequency']
        underused_gems = merged_df.nlargest(5, 'performance_ratio')
        
        st.success("**🔍 Underused High-Performers**")
        st.write("These keywords have high engagement but are used infrequently:")
        
        for _, row in underused_gems.iterrows():
            st.write(f"• **{row['Keyword']}**: {row['Avg_Engagement']:.0f} avg engagement, only used {row['Frequency']} times")
        
        # Engagement distribution
        percentiles = st.session_state.classifier.get_engagement_percentiles()
        if percentiles:
            st.subheader("📊 Engagement Benchmarks")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("50th Percentile", f"{percentiles['50th']:.0f}")
                st.caption("Median engagement")
            
            with col2:
                st.metric("75th Percentile", f"{percentiles['75th']:.0f}")
                st.caption("Good engagement")
            
            with col3:
                st.metric("90th Percentile", f"{percentiles['90th']:.0f}")
                st.caption("Excellent engagement")

def main():
    st.markdown("<h1 class='main-header'>📊 Instagram Dictionary Classifier</h1>", unsafe_allow_html=True)
    st.markdown("### Analyze Instagram engagement based on personalized language patterns")
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📁 Data Upload", "🔍 Keyword Analysis", "🔮 Predictions", "💡 Insights"]
    )
    
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "🔍 Keyword Analysis":
        keyword_analysis_page()
    elif page == "🔮 Predictions":
        prediction_page()
    elif page == "💡 Insights":
        insights_page()

if __name__ == "__main__":
    main()
