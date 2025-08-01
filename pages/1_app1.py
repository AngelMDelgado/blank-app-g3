import streamlit as st
import pandas as pd
import re
import io

# Set page config
st.set_page_config(
    page_title="📘 PA6 Method 2 – Classifier Word Metrics App",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📘 PA6 Method 2 – Classifier Word Metrics App")
st.markdown("**Team Project Notebook – Personalized Style**")
st.markdown("---")

# Initialize session state for results
if 'df_grouped' not in st.session_state:
    st.session_state.df_grouped = None
if 'final_df' not in st.session_state:
    st.session_state.final_df = None

# ---------------------------------------------
# SECTION 1: Upload Required Files
# ---------------------------------------------
st.header("📂 Section 1: Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload 'ig_posts_transformed.csv'", 
    type=['csv'],
    help="Upload your Instagram posts dataset with Caption and ID columns"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Show preview
        with st.expander("📋 Preview uploaded data"):
            st.dataframe(df.head())
        
        # Check required columns
        required_cols = ['Caption', 'ID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()
        
        # ---------------------------------------------
        # SECTION 2: Define Keyword Dictionary
        # ---------------------------------------------
        st.header("🔤 Section 2: Customize Keyword Dictionary")
        
        # Default keywords
        default_keywords = [
            "personalized", "custom", "tailored", "human", "experience",
            "connection", "care", "thoughtful", "responsive", "concierge"
        ]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Edit Keywords")
            keywords_text = st.text_area(
                "Keywords (one per line):",
                value="\n".join(default_keywords),
                height=200,
                help="Enter keywords that represent your brand language, one per line"
            )
            keywords = [kw.strip().lower() for kw in keywords_text.split('\n') if kw.strip()]
        
        with col2:
            st.subheader("Current Keywords")
            for i, kw in enumerate(keywords, 1):
                st.write(f"{i}. {kw}")
        
        # Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Reset to Default"):
                st.rerun()
        
        with col2:
            process_btn = st.button("🚀 Process Data", type="primary")
        
        # ---------------------------------------------
        # SECTION 3: Count Matched Words Per Statement
        # ---------------------------------------------
        if process_btn and keywords:
            st.header("📊 Section 3: Word Matching Analysis")
            
            def count_matches(caption):
                # Lowercase and remove punctuation using regex
                words = re.findall(r'\b\w+\b', str(caption).lower())
                matched = sum(1 for word in words if any(k in word for k in keywords))  # partial match
                return pd.Series([len(words), matched])
            
            # Apply the function
            with st.spinner("Processing word matches..."):
                df[['total_words', 'matched_words']] = df['Caption'].apply(count_matches)
                
                # Calculate match percentage
                df['match_pct'] = df['matched_words'] / df['total_words']
                df['match_pct'] = df['match_pct'].fillna(0)  # Handle division by zero
            
            st.success("✅ Word matching completed!")
            
            # Preview results
            st.subheader("📋 Sample keyword matches:")
            preview_df = df[['ID', 'Caption', 'matched_words', 'total_words', 'match_pct']].head(10)
            st.dataframe(preview_df)
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Posts", len(df))
            with col2:
                st.metric("Avg Matched Words", f"{df['matched_words'].mean():.2f}")
            with col3:
                st.metric("Avg Total Words", f"{df['total_words'].mean():.2f}")
            with col4:
                st.metric("Avg Match %", f"{df['match_pct'].mean():.2%}")
            
            # ---------------------------------------------
            # SECTION 4: Aggregate to IG Post Level
            # ---------------------------------------------
            st.header("📦 Section 4: Aggregate to Post Level")
            
            # Group by ID and sum matched/total words
            df_grouped = df.groupby('ID').agg({
                'matched_words': 'sum',
                'total_words': 'sum'
            }).reset_index()
            
            # Recalculate match_pct
            df_grouped['match_pct'] = df_grouped['matched_words'] / df_grouped['total_words']
            df_grouped['match_pct'] = df_grouped['match_pct'].fillna(0)
            
            st.success("✅ Data aggregated to post level!")
            st.dataframe(df_grouped.head(10))
            
            # Store in session state
            st.session_state.df_grouped = df_grouped
            
            # Download aggregated results
            csv_buffer = io.StringIO()
            df_grouped.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Aggregated Results",
                data=csv_buffer.getvalue(),
                file_name="ig_posts_aggregated.csv",
                mime="text/csv"
            )

# ---------------------------------------------
# SECTION 5: Upload & Join Engagement Data
# ---------------------------------------------
st.header("📈 Section 5: Join Engagement Data (Optional)")

if st.session_state.df_grouped is not None:
    engagement_file = st.file_uploader(
        "Upload 'ig_posts_shi_new.csv' (engagement data)", 
        type=['csv'],
        help="Upload engagement data with shortcode, number_likes, and number_comments columns"
    )
    
    if engagement_file is not None:
        try:
            df_engage = pd.read_csv(engagement_file)
            st.success(f"✅ Engagement file uploaded! Shape: {df_engage.shape}")
            
            with st.expander("📋 Preview engagement data"):
                st.dataframe(df_engage.head())
            
            # Check required columns
            required_engage_cols = ['shortcode', 'number_likes', 'number_comments']
            missing_engage_cols = [col for col in required_engage_cols if col not in df_engage.columns]
            
            if missing_engage_cols:
                st.error(f"❌ Missing required columns in engagement data: {missing_engage_cols}")
            else:
                # Merge data
                df_merged = pd.merge(
                    st.session_state.df_grouped, 
                    df_engage, 
                    left_on='ID', 
                    right_on='shortcode',
                    how='inner'
                )
                
                # Select relevant columns
                final_df = df_merged[['ID', 'match_pct', 'number_likes', 'number_comments']]
                
                st.success("✅ Data merged successfully!")
                st.subheader("📋 Final merged table:")
                st.dataframe(final_df.head(10))
                
                # Store in session state
                st.session_state.final_df = final_df
                
                # ---------------------------------------------
                # SECTION 6: Correlation Analysis
                # ---------------------------------------------
                st.header("📈 Section 6: Correlation Analysis")
                
                likes_corr = final_df['match_pct'].corr(final_df['number_likes'])
                comments_corr = final_df['match_pct'].corr(final_df['number_comments'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "📈 Match % vs Likes Correlation", 
                        f"{likes_corr:.3f}",
                        help="Correlation between keyword match percentage and number of likes"
                    )
                with col2:
                    st.metric(
                        "💬 Match % vs Comments Correlation", 
                        f"{comments_corr:.3f}",
                        help="Correlation between keyword match percentage and number of comments"
                    )
                
                # Interpretation
                st.subheader("🔍 Interpretation")
                if abs(likes_corr) > 0.3:
                    likes_strength = "Strong" if abs(likes_corr) > 0.5 else "Moderate"
                    likes_direction = "positive" if likes_corr > 0 else "negative"
                    st.write(f"📈 **Likes**: {likes_strength} {likes_direction} correlation ({likes_corr:.3f})")
                else:
                    st.write(f"📈 **Likes**: Weak correlation ({likes_corr:.3f})")
                
                if abs(comments_corr) > 0.3:
                    comments_strength = "Strong" if abs(comments_corr) > 0.5 else "Moderate"
                    comments_direction = "positive" if comments_corr > 0 else "negative"
                    st.write(f"💬 **Comments**: {comments_strength} {comments_direction} correlation ({comments_corr:.3f})")
                else:
                    st.write(f"💬 **Comments**: Weak correlation ({comments_corr:.3f})")
                
                # ---------------------------------------------
                # SECTION 7: Export Results
                # ---------------------------------------------
                st.header("💾 Section 7: Export Results")
                
                # Download final results
                final_csv_buffer = io.StringIO()
                final_df.to_csv(final_csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Final Results (IG_joined_result.csv)",
                    data=final_csv_buffer.getvalue(),
                    file_name="IG_joined_result.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                with st.expander("📊 Summary Statistics"):
                    st.write("**Dataset Summary:**")
                    st.write(f"- Total posts analyzed: {len(final_df)}")
                    st.write(f"- Average keyword match rate: {final_df['match_pct'].mean():.2%}")
                    st.write(f"- Average likes per post: {final_df['number_likes'].mean():.0f}")
                    st.write(f"- Average comments per post: {final_df['number_comments'].mean():.1f}")
                    
                    st.write("**Correlation Summary:**")
                    st.write(f"- Keyword match % vs Likes: {likes_corr:.3f}")
                    st.write(f"- Keyword match % vs Comments: {comments_corr:.3f}")
        
        except Exception as e:
            st.error(f"❌ Error processing engagement file: {str(e)}")

else:
    st.info("👆 Please process the main dataset first before uploading engagement data.")

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.markdown("**📘 PA6 Method 2 – Classifier Word Metrics App**")
st.markdown("*Upload your data, customize keywords, and analyze engagement correlations*")
