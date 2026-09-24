import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing


from utils.validator import validate_schema
from utils.preprocessor import calculate_rfm, prepare_time_series_data


# ============================================================================
# PAGE CONFIGURATION & CUSTOM CSS
# ============================================================================
st.set_page_config(
    page_title="Retail Analytics Intelligence Hub",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, attractive UI
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Alternative metric card colors */
    .metric-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-card.green {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    
    .metric-card.orange {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .metric-card.purple {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #1a1a2e;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #f0e6ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1.5rem 0;
    }
    
    .info-box strong {
        color: #553c9a;
    }
    
    /* File uploader container */
    .uploader-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Tab styling enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Dataframe styling */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        padding: 1rem;
    }
    
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Success/Error message styling */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR - Navigation & Info
# ============================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-header">🎯 Navigation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 1rem 0;">
        <div style="margin-bottom: 1rem;">
            <strong style="color: #4a5568;">📊 Analytics Features:</strong>
        </div>
        <div style="padding-left: 1rem;">
            • Customer Segmentation (RFM)<br>
            • Sales Forecasting<br>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 1rem 0;">
        <strong style="color: #4a5568;">ℹ️ About</strong>
        <p style="font-size: 0.9rem; color: #718096; margin-top: 0.5rem;">
            Advanced retail analytics platform for customer insights and revenue forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.info("""
    **Supported Formats:**
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    
    **Required Columns:**
    - Customer ID
    - Order Date
    - Sales Amount
    """)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header Section
st.markdown('<h1 class="main-title">🛍️ Retail Analytics Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your retail transaction dataset to perform <strong>Customer Segmentation (RFM)</strong> and <strong>Sales Forecasting</strong></p>', unsafe_allow_html=True)


# File Upload Section
uploaded_file = st.file_uploader(
    "📁 Upload your sales dataset (CSV or Excel)",
    type=["csv", "xlsx"],
    help="Ensure your file contains: Customer ID, Order Date, and Sales Amount columns"
)


if uploaded_file is not None:
    # Load dataset
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()
    
    # Success message
    st.markdown('<div class="success-box">✅ <strong>File uploaded successfully!</strong> Ready to analyze your data.</div>', unsafe_allow_html=True)
    
    # Data Preview in Expander
    with st.expander("🔍 Preview Raw Data (First 5 Rows)", expanded=False):
        st.dataframe(df.head(), use_container_width=True, height=200)
    
    # Validate Schema
    is_valid, message = validate_schema(df)
    if not is_valid:
        st.error(f"❌ **Schema Validation Failed:** {message}")
        st.stop()
    else:
        st.markdown(f'<div class="info-box">✅ <strong>Schema Validated:</strong> {message}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Create Tabs for different features
    tab1, tab2 = st.tabs(["👥 Customer Segmentation (RFM)", "📈 Sales Forecasting"])
    
    
    # ============================================================================
    # TAB 1: CUSTOMER SEGMENTATION
    # ============================================================================
    with tab1:
        st.markdown('<h2 class="section-header">Customer Value & Behavior Clusters</h2>', unsafe_allow_html=True)
        
        with st.spinner("⏳ Processing RFM metrics and clustering customers..."):
            rfm_df = calculate_rfm(df)
            
            # Scale and cluster
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
            
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(rfm_scaled)
            rfm_df['Cluster'] = clusters
            
            # Map cluster labels to meaningful business tiers
            cluster_summary = rfm_df.groupby('Cluster')['Monetary'].mean().reset_index()
            cluster_summary = cluster_summary.sort_values(by='Monetary', ascending=True).reset_index(drop=True)
            
            tier_mapping = {
                cluster_summary.loc[0, 'Cluster']: 'At-Risk / Low Value',
                cluster_summary.loc[1, 'Cluster']: 'Potential Loyalists',
                cluster_summary.loc[2, 'Cluster']: 'Loyal Customers',
                cluster_summary.loc[3, 'Cluster']: 'Champions 👑'
            }
            rfm_df['Customer Segment'] = rfm_df['Cluster'].map(tier_mapping)
        
        st.markdown("")  # Spacing
        
        # Metrics Row with Enhanced Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Customers</h3>
                <div class="value">{len(rfm_df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card blue">
                <h3>Avg Monetary Value</h3>
                <div class="value">${rfm_df['Monetary'].mean():,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card green">
                <h3>Avg Frequency</h3>
                <div class="value">{rfm_df['Frequency'].mean():.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card orange">
                <h3>Avg Recency</h3>
                <div class="value">{rfm_df['Recency'].mean():.0f}d</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")  # Spacing
        
        # Scatter Plot
        fig_rfm = px.scatter(
            rfm_df, 
            x="Recency", 
            y="Monetary", 
            size="Frequency", 
            color="Customer Segment",
            hover_name="Customer ID",
            title="Customer Segments: Recency vs. Monetary Value (Size = Frequency)",
            log_y=True,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_rfm.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', size=12),
            title_font=dict(size=16, color="#cacfda"),
            legend_title_font=dict(size=14),
            legend_font=dict(size=12),
            margin=dict(l=60, r=40, t=80, b=60),
            height=500
        )
        st.plotly_chart(fig_rfm, use_container_width=True)
        
        # Segment Table & Download
        st.markdown('<h3 class="section-header" style="font-size: 1.3rem; margin-top: 2rem;">📋 Segment Breakdown</h3>', unsafe_allow_html=True)
        segment_counts = rfm_df['Customer Segment'].value_counts().reset_index()
        segment_counts.columns = ['Customer Segment', 'Customer Count']
        st.dataframe(segment_counts, use_container_width=True, height=250)
        
        st.markdown("")  # Spacing
        
        # Download Button
        csv_data_rfm = rfm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Segmented Customer Data (CSV)",
            data=csv_data_rfm,
            file_name="segmented_retail_customers.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    
    # ============================================================================
    # TAB 2: SALES FORECASTING
    # ============================================================================
    with tab2:
        st.markdown('<h2 class="section-header">Future Revenue & Sales Trends</h2>', unsafe_allow_html=True)
        
        # Forecast slider in a styled container
        forecast_days = st.slider(
            "Select number of days to forecast into the future:",
            min_value=7,
            max_value=90,
            value=30,
            step=1,
            help="Choose how many days ahead you want to predict sales"
        )
        
        with st.spinner("⏳ Fitting Holt-Winters exponential smoothing model..."):
            ts_df = prepare_time_series_data(df)
            
            # Fit Exponential Smoothing Model (Holt-Winters)
            try:
                model = ExponentialSmoothing(
                    ts_df['Sales'], 
                    trend='add', 
                    seasonal=None, 
                    use_boxcox=False
                ).fit()
                
                forecast_values = model.forecast(forecast_days)
                
                # Generate future dates index
                last_date = ts_df['Order Date'].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
                
                forecast_df = pd.DataFrame({
                    'Order Date': future_dates,
                    'Forecasted Sales': forecast_values
                })
            except Exception as e:
                st.error(f"❌ Time series model error (insufficient or erratic date points): {e}")
                st.stop()
        
        st.markdown("")  # Spacing
        
        # Combine historical and forecasted for plotting
        ts_df['Type'] = 'Historical Sales'
        forecast_df['Type'] = 'Forecasted Sales'
        forecast_df['Sales'] = forecast_df['Forecasted Sales']
        
        plot_df = pd.concat([
            ts_df[['Order Date', 'Sales', 'Type']],
            forecast_df[['Order Date', 'Sales', 'Type']]
        ])
        
        # Forecasting Chart
        fig_ts = px.line(
            plot_df, 
            x='Order Date', 
            y='Sales', 
            color='Type',
            title=f"Historical vs. Next {forecast_days} Days Sales Forecast",
            color_discrete_map={'Historical Sales': '#1f77b4', 'Forecasted Sales': '#ff7f0e'}
        )
        fig_ts.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', size=12),
            title_font=dict(size=16, color="#eaeff8"),
            legend_title_font=dict(size=14),
            legend_font=dict(size=12),
            margin=dict(l=60, r=40, t=80, b=60),
            height=500,
            xaxis=dict(
                gridcolor='#e2e8f0',
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                gridcolor='#e2e8f0',
                tickfont=dict(size=11)
            )
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        st.markdown("")  # Spacing
        
        # Forecast Summary Metrics
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3>Projected Total Revenue</h3>
                <div class="value">${forecast_df['Forecasted Sales'].sum():,.0f}</div>
                <div style="font-size: 0.8rem; opacity: 0.9; margin-top: 0.5rem;">(Forecast Period)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_f2:
            st.markdown(f"""
            <div class="metric-card green">
                <h3>Projected Daily Average</h3>
                <div class="value">${forecast_df['Forecasted Sales'].mean():,.0f}</div>
                <div style="font-size: 0.8rem; opacity: 0.9; margin-top: 0.5rem;">(Per Day Sales)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")  # Spacing
        
        # Download Forecast Data
        csv_data_ts = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sales Forecast Data (CSV)",
            data=csv_data_ts,
            file_name="sales_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )


else:
    # Empty state with attractive design
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 12px; margin: 3rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
        <h2 style="color: #2d3748; margin-bottom: 1rem;">Ready to Analyze Your Retail Data?</h2>
        <p style="color: #4a5568; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
            Upload your CSV or Excel file to unlock powerful customer segmentation and sales forecasting insights.
        </p>
        <div style="display: inline-block; padding: 1rem 2rem; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <strong style="color: #667eea;">👆 Use the file uploader above to get started</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)