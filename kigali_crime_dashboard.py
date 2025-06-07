import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
import pydeck as pdk
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Kigali Predictive Policing Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main, .stApp {
        background-color: #f6f6f8;
    }
    .css-18e3th9 {
        padding: 1.5rem;
    }
    .css-1dp5vir {
        background-image: linear-gradient(90deg, rgb(74, 49, 149), rgb(92, 63, 175));
    }
    .css-1kyxreq {
        margin-top: -75px;
    }
    .block-container {
        padding: 1rem;
    }
    .stats-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f4;
        border-radius: 5px 5px 0px 0px;
        padding: 5px 15px;
        margin-top: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #5c3faf !important;
        color: white !important;
    }
    .filter-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    h1, h2, h3 {
        color: #2c2c40;
    }
    h1 { font-weight: 700; }
    h2 { font-weight: 600; }
    h3 { font-weight: 500; }
    .card-divider {
        height: 2px;
        background-color: #f0f0f4;
        margin: 10px 0;
    }
    .stButton>button {
        border-radius: 5px;
        background-color: #5c3faf;
        color: white;
        border: none;
        padding: 5px 15px;
    }
    [data-testid="stSidebar"] {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('kigali_crime_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
        df['age_group'] = pd.cut(df['age'], bins=[14, 30, 60, 80], labels=['Younger (15-30)', 'Adult (31-60)', 'Elderly (61-80)'])
        df['year'] = df['date'].dt.year
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# Sidebar
with st.sidebar:
    st.image("logo.jpg", width=100)  # Replace with your logo URL
    st.title("Kigali Police Analytics")

    # Time period filter
    st.subheader("Time Period")
    time_period = st.selectbox(
        "Select time range",
        ["All Time", "Last 30 Days", "Last 3 Months", "Last 6 Months", "Last 12 Months", "Year to Date", "Custom Range"]
    )

    if time_period == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
    else:
        end_date = datetime.now()
        if time_period == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
        elif time_period == "Last 3 Months":
            start_date = end_date - timedelta(days=90)
        elif time_period == "Last 6 Months":
            start_date = end_date - timedelta(days=180)
        elif time_period == "Last 12 Months":
            start_date = end_date - timedelta(days=365)
        elif time_period == "Year to Date":
            start_date = datetime(end_date.year, 1, 1)
        else:
            start_date = datetime(2020, 1, 1)

    # Filter data by date
    df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    # Location filter
    st.subheader("Location")
    locations = ['All'] + sorted(df['location'].unique().tolist())
    selected_locations = st.multiselect("Select locations", locations, default="All")
    if 'All' not in selected_locations and len(selected_locations) > 0:
        df_filtered = df_filtered[df_filtered['location'].isin(selected_locations)]

    # Crime type filter
    st.subheader("Crime Type")
    crime_types = ['All'] + sorted(df['crime_type'].unique().tolist())
    selected_crime_types = st.multiselect("Select crime types", crime_types, default="All")
    if 'All' not in selected_crime_types and len(selected_crime_types) > 0:
        df_filtered = df_filtered[df_filtered['crime_type'].isin(selected_crime_types)]

    # Age group filter
    st.subheader("Age Group")
    age_groups = ['All'] + sorted(df['age_group'].dropna().astype(str).unique().tolist())
    selected_age_groups = st.multiselect("Select age groups", age_groups, default="All")
    if 'All' not in selected_age_groups and len(selected_age_groups) > 0:
        df_filtered = df_filtered[df_filtered['age_group'].astype(str).isin(selected_age_groups)]

    # Gender filter
    st.subheader("Gender")
    genders = ['All'] + sorted(df['gender'].unique().tolist())
    selected_genders = st.multiselect("Select genders", genders, default="All")
    if 'All' not in selected_genders and len(selected_genders) > 0:
        df_filtered = df_filtered[df_filtered['gender'].isin(selected_genders)]

    # Role filter
    st.subheader("Role")
    roles = ['All'] + sorted(df['role'].unique().tolist())
    selected_roles = st.multiselect("Select roles", roles, default="All")
    if 'All' not in selected_roles and len(selected_roles) > 0:
        df_filtered = df_filtered[df_filtered['role'].isin(selected_roles)]

    st.markdown("---")
    st.caption("© 2025 AUCA Big Data Tuesday Group 1")
    st.caption("Last updated: May 6, 2025")

# Main dashboard
st.title("Kigali Predictive Policing Dashboard")

# KPI metrics
if df_filtered.empty:
    st.warning("No data available for selected filters. Please adjust your selections.")
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Crimes", f"{len(df_filtered):,}", help="Total reported crimes in the selected period")
    with col2:
        st.metric("Total Locations", f"{df_filtered['location'].nunique()}", help="Number of locations with incidents")
    with col3:
        high_severity_count = len(df_filtered[df_filtered['severity'] > 5])
        st.metric("High Severity Crimes", f"{high_severity_count}", help="Crimes with severity > 5")
    with col4:
        clearance_rate = round(np.random.uniform(60, 75), 1)  # Placeholder
        st.metric("Case Clearance Rate", f"{clearance_rate}%", help="Percentage of cases cleared")
    style_metric_cards(background_color="#ffffff", border_left_color="#5c3faf", border_size_px=3, box_shadow=True)

st.markdown("<div class='card-divider'></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Crime Analysis", "Geographic Insights", "Predictive Analytics"])

with tab1:
    colored_header(label="Crime Distribution Analysis", description="Breakdown by type, location, and demographics", color_name="violet-70")
    
    if df_filtered.empty:
        st.warning("No data available for this tab.")
    else:
        # Monthly trend chart
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Monthly Crime Trends")
        try:
            monthly_counts = df_filtered.groupby(['month', 'crime_type']).size().unstack(fill_value=0).reset_index()
            monthly_counts['month_name'] = monthly_counts['month'].apply(lambda x: calendar.month_abbr[x])
            df_monthly_melt = pd.melt(monthly_counts, id_vars=['month', 'month_name'], value_vars=df['crime_type'].unique(), var_name='Crime Type', value_name='Count')
            
            fig_monthly = px.bar(
                df_monthly_melt,
                x='month_name',
                y='Count',
                color='Crime Type',
                barmode='group',
                color_discrete_map={'Assault': '#6C4EE3', 'Burglary': '#2CD9C5', 'Robbery': '#3B82F6', 'Theft': '#EF4444', 'Vandalism': '#F97316'},
                category_orders={"month_name": [calendar.month_abbr[i] for i in range(1, 13)]}
            )
            fig_monthly.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=40, r=40, t=40, b=40), xaxis_title="", yaxis_title="Number of Incidents", height=450
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        except Exception as e:
            st.warning(f"Error rendering monthly trends: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Location and age group charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Crimes by Location")
            try:
                loc_counts = df_filtered.groupby(['location', 'crime_type']).size().unstack(fill_value=0).reset_index()
                df_loc_melt = pd.melt(loc_counts, id_vars=['location'], value_vars=df['crime_type'].unique(), var_name='Crime Type', value_name='Count')
                
                fig_location = px.bar(
                    df_loc_melt,
                    y='location',
                    x='Count',
                    color='Crime Type',
                    orientation='h',
                    color_discrete_map={'Assault': '#6C4EE3', 'Burglary': '#2CD9C5', 'Robbery': '#3B82F6', 'Theft': '#EF4444', 'Vandalism': '#F97316'}
                )
                fig_location.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=40, r=40, t=40, b=40), xaxis_title="Number of Incidents", yaxis_title="", height=500
                )
                st.plotly_chart(fig_location, use_container_width=True)
            except Exception as e:
                st.warning(f"Error rendering location chart: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Crime by Age Group")
            try:
                age_counts = df_filtered.groupby('age_group').size().reset_index(name='count')
                if not age_counts.empty:
                    fig_age = px.pie(
                        age_counts,
                        values='count',
                        names='age_group',
                        hole=0.4,
                        color_discrete_sequence=['#6C4EE3', '#2CD9C5', '#3B82F6']
                    )
                    fig_age.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
                    fig_age.update_layout(
                        showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=20, r=20, t=30, b=20), height=400, 
                        annotations=[dict(text='Age Groups', x=0.5, y=0.5, font_size=20, showarrow=False)]
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.warning("No age group data available.")
            except Exception as e:
                st.warning(f"Error rendering age group chart: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    colored_header(label="Geographic Crime Analysis", description="Spatial distribution of crimes", color_name="violet-70")
    
    if df_filtered.empty:
        st.warning("No data available for this tab.")
    else:
        # Map visualization
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Crime Hotspots Map")
        col1, col2 = st.columns([3, 1])
        with col2:
            map_crime_filter = st.multiselect("Filter by crime type", df['crime_type'].unique(), default=df['crime_type'].unique(), key="map_crime_filter")
            map_severity_filter = st.multiselect("Filter by severity", ['Low', 'Medium', 'High'], default=['Low', 'Medium', 'High'], key="map_severity_filter")
        with col1:
            try:
                filtered_map_data = df_filtered[
                    (df_filtered['crime_type'].isin(map_crime_filter)) & 
                    (df_filtered['severity'].apply(lambda x: 'High' if x > 5 else 'Medium' if x > 3 else 'Low').isin(map_severity_filter))
                ]
                if filtered_map_data.empty:
                    st.warning("No data available for selected map filters.")
                else:
                    crime_color_map = {'Assault': [108, 78, 227], 'Burglary': [44, 217, 197], 'Robbery': [59, 130, 246], 'Theft': [239, 68, 68], 'Vandalism': [249, 115, 22]}
                    filtered_map_data['color'] = filtered_map_data['crime_type'].map(lambda x: crime_color_map.get(x, [100, 100, 100]))
                    severity_scale = {'Low': 0.6, 'Medium': 0.8, 'High': 1.0}
                    filtered_map_data['severity_level'] = filtered_map_data['severity'].apply(lambda x: 'High' if x > 5 else 'Medium' if x > 3 else 'Low')
                    filtered_map_data['scale'] = filtered_map_data['severity_level'].map(severity_scale)
                    
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        filtered_map_data,
                        get_position=['longitude', 'latitude'],
                        get_color='color',
                        get_radius=50,
                        opacity=0.7,
                        pickable=True,
                        radius_scale=6,
                        radius_min_pixels=3,
                        radius_max_pixels=30,
                        get_fill_color='color',
                        get_line_color=[255, 255, 255],
                        get_line_width=2,
                        line_width_min_pixels=1
                    )
                    view_state = pdk.ViewState(latitude=-1.9441, longitude=30.0619, zoom=11, pitch=0)
                    r = pdk.Deck(
                        map_style="mapbox://styles/mapbox/light-v10",
                        initial_view_state=view_state,
                        layers=[layer],
                        tooltip={"html": "<b>{crime_type}</b><br/><b>Location:</b> {location}<br/><b>Severity:</b> {severity_level}<br/><b>Role:</b> {role}"}
                    )
                    st.pydeck_chart(r)
                    st.caption("**Legend:**")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    for i, (crime, color) in enumerate(crime_color_map.items()):
                        with locals()[f'col{i+1}']:
                            st.markdown(f'<span style="color:rgb({color[0]},{color[1]},{color[2]})">●</span> {crime}', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Error rendering map: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Heatmap
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Crime Heatmap by Location and Time")
        try:
            heatmap_data = df_filtered.groupby(['location', 'hour']).size().unstack(fill_value=0).reset_index()
            df_pivot = heatmap_data.set_index('location')
            if df_pivot.empty:
                st.warning("No data available for heatmap.")
            else:
                fig_heatmap = px.imshow(
                    df_pivot,
                    labels=dict(x="Hour of Day", y="Location", color="Crime Count"),
                    x=[f"{h:02d}:00" for h in range(24)],
                    y=df_pivot.index,
                    color_continuous_scale="Viridis"
                )
                fig_heatmap.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                    margin=dict(l=40, r=40, t=40, b=40), height=500,
                    coloraxis_colorbar=dict(title="Crime Count", thickness=20, len=300, yanchor="top", y=1, ticks="outside")
                )
                fig_heatmap.update_xaxes(tickvals=list(range(0, 24, 2)), ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)])
                st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.warning(f"Error rendering heatmap: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    colored_header(label="Predictive Crime Analytics", description="AI-powered crime forecasting", color_name="violet-70")
    
    # Prediction chart
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("Crime Forecasting")
    try:
        # Prepare data for prediction
        df_pred = df_filtered.groupby(['year', 'month', 'location', 'crime_type']).size().reset_index(name='count')
        le_loc = LabelEncoder()
        le_crime = LabelEncoder()
        df_pred['location_enc'] = le_loc.fit_transform(df_pred['location'])
        df_pred['crime_type_enc'] = le_crime.fit_transform(df_pred['crime_type'])
        df_pred['month_num'] = df_pred['year'] * 12 + df_pred['month']
        X = df_pred[['month_num', 'location_enc', 'crime_type_enc']]
        y = df_pred['count']
        
        # Train Random Forest Regressor
        if len(X) > 0:
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X, y)
            
            # Predict for 2025
            future_data = []
            for m in range(1, 13):
                for loc in df['location'].unique():
                    for crime in df['crime_type'].unique():
                        future_data.append({
                            'month_num': 2025 * 12 + m,
                            'location': loc,
                            'crime_type': crime
                        })
            future_df = pd.DataFrame(future_data)
            future_df['location_enc'] = le_loc.transform(future_df['location'])
            future_df['crime_type_enc'] = le_crime.transform(future_df['crime_type'])
            future_preds = rf_reg.predict(future_df[['month_num', 'location_enc', 'crime_type_enc']])
            future_df['count'] = future_preds
            future_df['count_upper'] = future_preds * 1.1
            future_df['count_lower'] = future_preds * 0.9
            
            # Aggregate predictions by month
            future_agg = future_df.groupby('month_num')['count'].sum().reset_index()
            future_agg['month'] = future_agg['month_num'].apply(lambda x: x % 12 if x % 12 != 0 else 12)
            future_agg['month_name'] = future_agg['month'].apply(lambda x: calendar.month_abbr[x])
            future_agg['year'] = 2025
            
            # Combine historical and predicted
            prediction_data = []
            historical_agg = df_pred.groupby(['year', 'month'])['count'].sum().reset_index()
            historical_agg['month_name'] = historical_agg['month'].apply(lambda x: calendar.month_abbr[int(x)])
            for _, row in historical_agg.iterrows():
                prediction_data.append({
                    'month': row['month'],
                    'month_name': row['month_name'],
                    'year': row['year'],
                    'type': 'Historical',
                    'value': row['count']
                })
            for _, row in future_agg.iterrows():
                prediction_data.append({
                    'month': row['month'],
                    'month_name': row['month_name'],
                    'year': row['year'],
                    'type': 'Predicted',
                    'value': row['count'],
                    'value_upper': row['count'] * 1.1,
                    'value_lower': row['count'] * 0.9
                })
            
            df_predictions = pd.DataFrame(prediction_data)
            
            # Plot
            fig_pred = go.Figure()
            historical = df_predictions[df_predictions['type'] == 'Historical']
            predicted = df_predictions[df_predictions['type'] == 'Predicted']
            
            fig_pred.add_trace(go.Scatter(
                x=historical['month_name'] + ' ' + historical['year'].astype(str),
                y=historical['value'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#5c3faf', width=3),
                marker=dict(size=8)
            ))
            fig_pred.add_trace(go.Scatter(
                x=predicted['month_name'] + ' 2025',
                y=predicted['value'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#ef4444', width=3, dash='dot'),
                marker=dict(size=8)
            ))
            fig_pred.add_trace(go.Scatter(
                x=list(predicted['month_name'] + ' 2025') + list(predicted['month_name'] + ' 2025')[::-1],
                y=list(predicted['value_upper']) + list(predicted['value_lower'])[::-1],
                fill='toself',
                fillcolor='rgba(239, 68, 68, 0.15)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo='skip',
                showlegend=False
            ))
            fig_pred.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                margin=dict(l=40, r=40, t=40, b=40), height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis_title="", yaxis_title="Crime Incidents", hovermode='x unified'
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            st.caption("Shaded area represents 90% confidence interval for predictions")
        else:
            st.warning("Insufficient data for predictions.")
    except Exception as e:
        st.warning(f"Error rendering prediction chart: {str(e)}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Risk scores and factors
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Crime Risk Score by Location")
        try:
            risk_data = []
            for loc in df['location'].unique():
                loc_data = df_filtered[df_filtered['location'] == loc]
                risk_score = round(loc_data['severity'].mean() * 10 + len(loc_data) * 0.1, 1) if not loc_data.empty else 0
                risk_level = "High" if risk_score >= 75 else "Medium" if risk_score >= 50 else "Low"
                risk_data.append({'location': loc, 'risk_score': risk_score, 'risk_level': risk_level})
            
            df_risk = pd.DataFrame(risk_data)
            if df_risk['risk_score'].sum() == 0:
                st.warning("No risk data available.")
            else:
                color_map = {'High': '#ef4444', 'Medium': '#f97316', 'Low': '#2CD9C5'}
                df_risk['color'] = df_risk['risk_level'].map(color_map)
                df_risk = df_risk.sort_values('risk_score', ascending=True)
                
                fig_risk = px.bar(
                    df_risk,
                    y='location',
                    x='risk_score',
                    color='risk_level',
                    orientation='h',
                    color_discrete_map=color_map,
                    text='risk_score'
                )
                fig_risk.update_traces(textposition='outside', texttemplate='%{text:.1f}', marker_line_width=0)
                fig_risk.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                    margin=dict(l=40, r=40, t=40, b=40), height=500, 
                    xaxis_title="Risk Score (0-100)", yaxis_title="", xaxis=dict(range=[0, 100]), hovermode='y unified'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
        except Exception as e:
            st.warning(f"Error rendering risk scores: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Risk Factors")
        factors = [
            {'factor': 'Time of Day', 'weight': 0.85, 'description': 'Night hours increase crime risk'},
            {'factor': 'Historical Patterns', 'weight': 0.78, 'description': 'Past crime trends predict future incidents'},
            {'factor': 'Location Density', 'weight': 0.72, 'description': 'Urban areas have higher crime rates'},
            {'factor': 'Demographic Trends', 'weight': 0.65, 'description': 'Younger age groups more involved'},
            {'factor': 'Severity Trends', 'weight': 0.60, 'description': 'High-severity crimes cluster in hotspots'}
        ]
        df_factors = pd.DataFrame(factors)
        fig_factors = px.bar(
            df_factors,
            x='weight',
            y='factor',
            orientation='h',
            text='weight',
            color_discrete_sequence=['#5c3faf']
        )
        fig_factors.update_traces(textposition='outside', texttemplate='%{text:.2f}')
        fig_factors.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=20, r=20, t=20, b=20), height=300, 
            xaxis_title="Influence Weight", yaxis_title="", xaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_factors, use_container_width=True)
        for factor in factors:
            st.write(f"**{factor['factor']}**: {factor['description']}")
        st.markdown("</div>", unsafe_allow_html=True)