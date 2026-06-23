import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Dashboard Ames Housing", layout="wide")

# Título y descripción
st.title("📊 Panel A: Análisis Exploratorio de Datos - Ames Housing")
st.markdown("Este panel permite explorar el dataset de forma interactiva y visualizar las relaciones clave.")

# Carga de datos (ajusta el nombre del archivo si es necesario)
@st.cache_data
def load_data():
    # Asegúrate de que el archivo dataset.csv esté en la misma carpeta que este script
    df = pd.read_csv('dataset.csv')
    return df

df = load_data()

# --- SIDEBAR: Filtros ---
st.sidebar.header("Filtros de Datos")

# Filtro por vecindario
neighborhoods = st.sidebar.multiselect(
    "Seleccionar Vecindarios", 
    options=df['Neighborhood'].unique(), 
    default=df['Neighborhood'].unique()
)

# Filtro por área habitable
min_area, max_area = int(df['GrLivArea'].min()), int(df['GrLivArea'].max())
area_range = st.sidebar.slider(
    "Rango de Área Habitable (sq ft)", 
    min_area, max_area, 
    (min_area, max_area)
)

# Aplicar filtros al dataframe
df_filtered = df[
    (df['Neighborhood'].isin(neighborhoods)) & 
    (df['GrLivArea'] >= area_range[0]) & 
    (df['GrLivArea'] <= area_range[1])
]

# --- KPIs superiores ---
col1, col2, col3 = st.columns(3)
col1.metric("Total de Registros", len(df_filtered))
col2.metric("Precio Promedio", f"${df_filtered['SalePrice'].mean():,.2f}")
col3.metric("Calidad Promedio", f"{df_filtered['OverallQual'].mean():.2f}/10")

st.markdown("---")

# --- Visualizaciones ---
col_a, col_b = st.columns(2)

with col_a:
    # 1. Distribución del precio
    fig1 = px.histogram(
        df_filtered, 
        x="SalePrice", 
        nbins=30, 
        title="Distribución de Precios de Venta", 
        color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    # 2. Dispersión: GrLivArea vs Price
    fig2 = px.scatter(
        df_filtered, 
        x="GrLivArea", 
        y="SalePrice", 
        color="OverallQual", 
        title="Relación Área Habitable vs Precio", 
        hover_data=['Neighborhood']
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3. Mapa de calor (Heatmap) de correlación
st.subheader("Mapa de Calor de Correlaciones")
numeric_df = df_filtered.select_dtypes(include=['number'])
fig3 = px.imshow(
    numeric_df.corr(), 
    title="Correlación entre variables numéricas", 
    color_continuous_scale='RdBu_r', 
    aspect="auto"
)
st.plotly_chart(fig3, use_container_width=True)