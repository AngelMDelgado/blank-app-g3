import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Instagram Personalized Language Analyzer",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📊 Instagram Personalized Language Analyzer</h1>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'final_df' not in st.session_state:
    st.session_state.final_df = None

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Default keywords
DEFAULT_KEYWORDS = [
    "personalized", "custom", "tailored", "bespoke", "individual",
    "human", "experience", "connection", "care", "thoughtful", 
    "responsive", "concierge", "attentive", "dedicated", "unique"
]

# Keyword customization
st.sidebar.subheader("🎯 Keyword Dictionary")
keyword_input_method = st.sidebar.radio(
    "Choose input method:",
    ["Use default keywords", "Upload keyword file", "Manual input"]
)

keywords = DEFAULT_KEYWORDS.copy()

if keyword_input_method == "Use default keywords":
    st.sidebar.write(f"Using {len(DEFAULT_KEYWORDS)} default keywords")
    with st.sidebar.expander("View default keywords"):
        st.write(", ".join(DEFAULT_KEYWORDS))

elif keyword_input_method == "Upload keyword file":
    keyword_file = st.sidebar.file_uploader(
        "Upload keyword file (one keyword per line)", 
        type=['txt', 'csv']
    )
    if keyword_file:
        try:
            if keyword_file.name.endswith('.txt'):
                keywords = keyword_file.read().decode('utf-8').strip().split('\n')
            else:
                df_keywords = pd.read_csv(keyword_file)
                keywords = df_keywords.iloc[:, 0].tolist()
            keywords = [k.strip().lower() for k in keywords if k.strip()]
            st.sidebar.success(f"Loaded {len(keywords)} keywords")
        except Exception as e:
            st.sidebar.error(f"Error loading keywords: {str(e)}")
            keywords = DEFAULT_KEYWORDS

elif keyword_input_method == "Manual input":
    keyword_text = st.sidebar.text_area(
        "Enter keywords (one per line):",
        value="\n".join(DEFAULT_KEYWORDS),
        height=200
    )
    if keyword_text:
        keywords = [k.strip().lower() for k in keyword_text.split('\n') if k.strip()]

# Analysis options
st.sidebar.subheader("⚙️ Analysis Options")
exact_match = st.sidebar.checkbox("Use exact word matching (recommended)", value=True)
min_word_count = st.sidebar.number_input("Minimum words per post", min_value=1, value=5)

# Helper functions
@st.cache_data
def count_keyword_matches(caption, keywords, exact_match=True):
    """Count keyword matches in caption with improved accuracy"""
    if pd.isna(caption):
        return pd.Series([0, 0, []])
    
    text = str(caption).lower()
    
    if exact_match:
        words = re.findall(r'\b\w+\b', text)
        matched_keywords = [word for word in words if word in keywords]
    else:
        words = re.findall(r'\b\w+\b', text)
        matched_keywords = [word for word in words if any(k in word for k in keywords)]
    
    matched_count = len(matched_keywords)
    total_words = len(words)
    
    return pd.Series([total_words, matched_count, matched_keywords])

def create_visualizations(final_df, likes_corr, comments_corr):
    """Create analysis visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Instagram Personalized Language Analysis', fontsize=16, fontweight='bold')
    
    # 1. Match percentage distribution
    axes[0,0].hist(final_df['match_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Personalized Language %')
    axes[0,0].set_xlabel('Match Percentage')
    axes[0,0].set_ylabel('Frequency')
    
    # 2. Scatter plot: Match % vs Likes
    axes[0,1].scatter(final_df['match_pct'], final_df['number_likes'], alpha=0.6, color='coral')
    axes[0,1].set_title(f'Personalized Language % vs Likes (r={likes_corr:.3f})')
    axes[0,1].set_xlabel('Match Percentage')
    axes[0,1].set_ylabel('Number of Likes')
    
    # Add trendline for likes
    if len(final_df) > 1:
        z = np.polyfit(final_df['match_pct'], final_df['number_likes'], 1)
        p = np.poly1d(z)
        axes[0,1].plot(final_df['match_pct'], p(final_df['match_pct']), "r--", alpha=0.8)
    
    # 3. Scatter plot: Match % vs Comments
    axes[1,0].scatter(final_df['match_pct'], final_df['number_comments'], alpha=0.6, color='lightgreen')
    axes[1,0].set_title(f'Personalized Language % vs Comments (r={comments_corr:.3f})')
    axes[1,0].set_xlabel('Match Percentage')
    axes[1,0].set_ylabel('Number of Comments')
    
    # Add trendline for comments
    if len(final_df) > 1:
        z2 = np.polyfit(final_df['match_pct'], final_df['number_comments'], 1)
        p2 = np.poly1d(z2)
        axes[1,0].plot(final_df['match_pct'], p2(final_df['match_pct']), "r--", alpha=0.8)
    
    # 4. Top performing posts by personalization
    top_posts = final_df.nlargest(min(10, len(final_df)), 'match_pct')
    if len(top_posts) > 0:
        axes[1,1].barh(range(len(top_posts)), top_posts['match_pct'], color='gold', alpha=0.7)
        axes[1,1].set_title(f'Top {len(top_posts)} Most Personalized Posts')
        axes[1,1].set_xlabel('Match Percentage')
        axes[1,1].set_ylabel('Post Rank')
        axes[1,1].invert_yaxis()
    
    plt.tight_layout()
    return fig

# Main app layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Data Upload")
    
    # File upload for main dataset
    uploaded_file1 = st.file_uploader(
        "Upload Instagram posts data (CSV)", 
        type=['csv'],
        key="main_data",
        help="Upload your 'ig_posts_transformed.csv' file or similar dataset with 'ID' and 'Statement' columns"
    )
    
    # File upload for engagement data
    uploaded_file2 = st.file_uploader(
        "Upload engagement data (CSV) - Optional", 
        type=['csv'],
        key="engagement_data",
        help="Upload your engagement data with 'shortcode', 'number_likes', and 'number_comments' columns"
    )

with col2:
    st.header("📊 Analysis Summary")
    if keywords:
        st.metric("Keywords to analyze", len(keywords))
        st.metric("Match type", "Exact" if exact_match else "Partial")
        st.metric("Min word count", min_word_count)

# Process uploaded data
if uploaded_file1 is not None:
    try:
        # Load main dataset
        df = pd.read_csv(uploaded_file1)
        
        st.success(f"✅ Loaded main dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate required columns
        required_cols = ['ID', 'Statement']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info("Required columns: ID, Statement")
            st.stop()
        
        # Show dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Unique Posts", df['ID'].nunique())
        with col3:
            st.metric("Missing Statements", df['Statement'].isna().sum())
        with col4:
            st.metric("Columns", len(df.columns))
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing personalized language..."):
                
                # Apply keyword matching
                results = df['Statement'].apply(
                    lambda x: count_keyword_matches(x, keywords, exact_match)
                )
                df[['total_words', 'matched_words', 'matched_keywords']] = results
                
                # Calculate match percentage
                df['match_pct'] = np.where(df['total_words'] >= min_word_count, 
                                          df['matched_words'] / df['total_words'], 
                                          0)
                
                # Filter by minimum word count
                df_filtered = df[df['total_words'] >= min_word_count].copy()
                
                # Aggregate to post level
                df_grouped = df_filtered.groupby('ID').agg({
                    'matched_words': 'sum',
                    'total_words': 'sum',
                    'matched_keywords': lambda x: list(set([item for sublist in x for item in sublist]))
                }).reset_index()
                
                df_grouped['match_pct'] = np.where(df_grouped['total_words'] > 0,
                                                  df_grouped['matched_words'] / df_grouped['total_words'],
                                                  0)
                
                # Merge with engagement data if available
                final_df = df_grouped
                
                if uploaded_file2 is not None:
                    try:
                        df_engage = pd.read_csv(uploaded_file2)
                        
                        # Check for required engagement columns
                        engagement_cols = ['shortcode', 'number_likes', 'number_comments']
                        if all(col in df_engage.columns for col in engagement_cols):
                            df_merged = pd.merge(df_grouped, df_engage, 
                                               left_on='ID', right_on='shortcode', 
                                               how='inner')
                            
                            final_df = df_merged[['ID', 'match_pct', 'matched_words', 'total_words', 
                                                 'number_likes', 'number_comments']].copy()
                            
                            st.success(f"✅ Merged with engagement data: {len(final_df)} posts")
                        else:
                            st.warning("⚠️ Engagement data missing required columns. Proceeding without engagement metrics.")
                    
                    except Exception as e:
                        st.warning(f"⚠️ Could not process engagement data: {str(e)}")
                
                # Store results in session state
                st.session_state.final_df = final_df
                st.session_state.analysis_complete = True
                
                st.success("🎉 Analysis completed!")

# Display results if analysis is complete
if st.session_state.analysis_complete and st.session_state.final_df is not None:
    final_df = st.session_state.final_df
    
    st.header("📈 Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Posts Analyzed", len(final_df))
    with col2:
        posts_with_matches = (final_df['match_pct'] > 0).sum()
        st.metric("Posts with Keywords", posts_with_matches)
    with col3:
        avg_match_pct = final_df['match_pct'].mean()
        st.metric("Avg Match %", f"{avg_match_pct:.1%}")
    with col4:
        total_keywords = final_df['matched_words'].sum()
        st.metric("Total Keywords Found", int(total_keywords))
    
    # Correlation analysis (if engagement data is available)
    if 'number_likes' in final_df.columns and 'number_comments' in final_df.columns:
        st.subheader("🔗 Correlation Analysis")
        
        likes_corr = final_df['match_pct'].corr(final_df['number_likes'])
        comments_corr = final_df['match_pct'].corr(final_df['number_comments'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Correlation: Match% vs Likes", f"{likes_corr:.4f}")
        with col2:
            st.metric("Correlation: Match% vs Comments", f"{comments_corr:.4f}")
        
        # Interpretation
        if likes_corr > 0.3:
            st.success("🚀 Strong positive correlation with likes! Consider increasing personalized language.")
        elif likes_corr > 0.1:
            st.info("📊 Moderate correlation with likes. Consider A/B testing personalization strategies.")
        else:
            st.warning("📉 Weak correlation with likes in this dataset.")
        
        # Statistical significance test
        personalized_posts = final_df[final_df['match_pct'] > 0]['number_likes']
        non_personalized_posts = final_df[final_df['match_pct'] == 0]['number_likes']
        
        if len(personalized_posts) > 0 and len(non_personalized_posts) > 0:
            t_stat, p_value = stats.ttest_ind(personalized_posts, non_personalized_posts)
            
            st.subheader("📊 Statistical Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_likes_personalized = personalized_posts.mean()
                st.metric("Avg Likes (Personalized)", f"{avg_likes_personalized:.0f}")
            with col2:
                avg_likes_non_personalized = non_personalized_posts.mean()
                st.metric("Avg Likes (Non-personalized)", f"{avg_likes_non_personalized:.0f}")
            with col3:
                st.metric("T-test p-value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ Statistically significant difference in engagement!")
            else:
                st.info("ℹ️ No statistically significant difference found.")
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    if 'number_likes' in final_df.columns and 'number_comments' in final_df.columns:
        likes_corr = final_df['match_pct'].corr(final_df['number_likes'])
        comments_corr = final_df['match_pct'].corr(final_df['number_comments'])
        
        fig = create_visualizations(final_df, likes_corr, comments_corr)
        st.pyplot(fig)
    else:
        # Simple histogram if no engagement data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(final_df['match_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of Personalized Language %')
        ax.set_xlabel('Match Percentage')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Show top personalized posts
    if len(final_df) > 0:
        top_personalized = final_df.nlargest(min(10, len(final_df)), 'match_pct')
        st.write("🏆 Top Most Personalized Posts:")
        st.dataframe(top_personalized, use_container_width=True)
    
    # Export functionality
    st.subheader("💾 Export Results")
    
    # Prepare export data
    export_df = final_df.copy()
    export_df['analysis_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    export_df['keywords_analyzed'] = len(keywords)
    export_df['match_type'] = 'exact' if exact_match else 'partial'
    
    # Convert to CSV for download
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name=f"ig_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Summary report
        if 'number_likes' in final_df.columns:
            summary_text = f"""Instagram Personalized Language Analysis Summary
{'='*50}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Keywords Analyzed: {len(keywords)}
Total Posts: {len(final_df)}
Posts with Personalized Language: {(final_df['match_pct'] > 0).sum()}

Key Findings:
• Average Personalization Rate: {final_df['match_pct'].mean():.1%}
• Correlation with Likes: {likes_corr:.4f}
• Correlation with Comments: {comments_corr:.4f}

Recommendation:
{'Strong positive correlation - increase personalized language!' if likes_corr > 0.3 else 
 'Moderate correlation - consider A/B testing personalization strategies.' if likes_corr > 0.1 else
 'Weak correlation - personalization may not drive engagement in this dataset.'}
"""
        else:
            summary_text = f"""Instagram Personalized Language Analysis Summary
{'='*50}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Keywords Analyzed: {len(keywords)}
Total Posts: {len(final_df)}
Posts with Personalized Language: {(final_df['match_pct'] > 0).sum()}

Key Findings:
• Average Personalization Rate: {final_df['match_pct'].mean():.1%}

Note: Upload engagement data for correlation analysis.
"""
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name=f"analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Instagram Personalized Language Analyzer</p>
    <p>Upload your data, customize keywords, and analyze the impact of personalized language on engagement.</p>
</div>
""", unsafe_allow_html=True)
