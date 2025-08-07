import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from collections import Counter, defaultdict

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
</style>
""", unsafe_allow_html=True)

class CategoryClassifier:
    def __init__(self):
        self.df = None
        self.categories = {}
        self.classification_results = None
        self.keyword_frequency = {}
        
    def load_data_from_uploads(self, posts_df, categories_dict=None):
        """Load data from Streamlit uploads"""
        try:
            self.df = posts_df.copy()
            
            required_cols = ['ID', 'Statement']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                st.error(f"Missing required columns in posts data: {missing_cols}")
                return False
            
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
            
            results.append({
                'ID': row['ID'],
                'Statement': row['Statement'],
                'Primary_Category': primary_category,
                'All_Categories': [cat['category'] for cat in post_categories],
                'Keywords_Found': found_keywords,
                'Keyword_Count': len(found_keywords)
            })
        
        self.classification_results = results
        self.keyword_frequency = dict(keyword_counts)
        
        return {
            'results': results,
            'category_counts': dict(category_counts),
            'keyword_frequency': dict(keyword_counts)
        }
    
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
    2. **Categories** (optional): JSON file or manual configuration
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
            success = st.session_state.classifier.load_data_from_uploads(
                st.session_state.posts_df,
                categories_dict
            )
            if success:
                st.session_state.data_loaded = True
                st.success("🎉 Data loaded successfully! Go to Category Analysis.")
        else:
            st.error("❌ Please upload posts data file.")

def category_analysis_page():
    st.markdown("<h2 class='section-header'>🔍 Category Analysis</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Upload section.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Current Data:**
        - Posts: {len(st.session_state.classifier.df)}
        - Categories: {len(st.session_state.classifier.categories)}
        """)
    
    with col2:
        if st.button("🔍 Analyze Categories", type="primary"):
            with st.spinner("Analyzing post categories..."):
                results = st.session_state.classifier.classify_posts()
                
                if results:
                    st.session_state.analysis_done = True
                    st.success("✅ Analysis completed!")
    
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
                selected_category = st.selectbox(
                    "Filter by category:",
                    ["All"] + list(st.session_state.classifier.categories.keys()) + ["Uncategorized"]
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
        ["📁 Data Upload", "🔍 Category Analysis"]
    )
    
    # Show current categories in sidebar
    if st.session_state.data_loaded:
        st.sidebar.markdown("### 🏷️ Current Categories")
        for category, keywords in st.session_state.classifier.categories.items():
            with st.sidebar.expander(category):
                st.write(", ".join(keywords))
    
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "🔍 Category Analysis":
        category_analysis_page()

if __name__ == "__main__":
    main()
