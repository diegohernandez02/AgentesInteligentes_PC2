import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Configuración de la página
st.set_page_config(page_title="Dashboard Ames Housing", layout="wide")

# Carga de datos y modelo
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df

@st.cache_resource
def load_model():
    return joblib.load('modelo_final.pkl')

df = load_data()
model = load_model()

# Título principal
st.title("🏡 Dashboard Inmobiliario - Ames Housing")

# Crear pestañas para los Paneles A y B
tab1, tab2 = st.tabs(["📊 Panel A: Análisis Exploratorio", "🔮 Panel B: Análisis Predictivo"])

with tab1:
    st.header("Análisis Exploratorio de Datos")
    
    # --- SIDEBAR para filtros ---
    st.sidebar.header("Filtros Panel A")
    neighborhoods = st.sidebar.multiselect("Seleccionar Vecindarios", options=df['Neighborhood'].unique(), default=df['Neighborhood'].unique())
    min_area, max_area = int(df['GrLivArea'].min()), int(df['GrLivArea'].max())
    area_range = st.sidebar.slider("Rango de Área Habitable (sq ft)", min_area, max_area, (min_area, max_area))

    df_filtered = df[(df['Neighborhood'].isin(neighborhoods)) & 
                     (df['GrLivArea'] >= area_range[0]) & 
                     (df['GrLivArea'] <= area_range[1])]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros", len(df_filtered))
    col2.metric("Precio Promedio", f"${df_filtered['SalePrice'].mean():,.2f}")
    col3.metric("Calidad Promedio", f"{df_filtered['OverallQual'].mean():.2f}/10")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.histogram(df_filtered, x="SalePrice", nbins=30, title="Distribución de Precios de Venta")
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = px.scatter(df_filtered, x="GrLivArea", y="SalePrice", color="OverallQual", title="Relación Área vs Precio")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("🔮 Panel B: Análisis Predictivo")
    st.write("Ingrese las características de la propiedad para obtener una estimación de precio.")
    
    # Formulario de entrada
    c1, c2 = st.columns(2)
    with c1:
        gr_liv_area = st.number_input("Área Habitable (sq ft)", 500, 5000, 1500)
        overall_qual = st.slider("Calidad General (1-10)", 1, 10, 5)
    with c2:
        neighborhood = st.selectbox("Vecindario", df['Neighborhood'].unique())
        total_bsmt_sf = st.number_input("Área de Sótano (sq ft)", 0, 3000, 800)

    if st.button("Calcular Predicción"):
        input_data = pd.DataFrame({
            'GrLivArea': [gr_liv_area],
            'OverallQual': [overall_qual],
            'Neighborhood': [neighborhood],
            'TotalBsmtSF': [total_bsmt_sf]
        })
        
        prediction = model.predict(input_data)
        
        st.success(f"### Precio estimado: ${prediction[0]:,.2f}")
        st.info("""
        **Contexto de la predicción:** Esta estimación se basa en los patrones de mercado aprendidos por nuestro modelo Random Forest. 
        El precio refleja la valoración estadística esperada dada la calidad y dimensiones de la vivienda.
        """)
