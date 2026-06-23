import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Configuración de la página
st.set_page_config(page_title="Dashboard Ames Housing", layout="wide")

# Lista de las 10 variables más influyentes (debe coincidir exactamente con el entrenamiento)
FEATURES = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 
            'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd', 'MasVnrArea']

# Carga de datos y modelo con caché
@st.cache_data
def load_data():
    return pd.read_csv('train.csv')

@st.cache_resource
def load_model():
    return joblib.load('modelo_final.pkl')

# Cargar elementos
try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar archivos: {e}. Asegúrate de que 'dataset.csv' y 'modelo_final.pkl' existan.")
    st.stop()

# Título principal
st.title("🏡 Dashboard Inmobiliario - Ames Housing")

# Crear pestañas
tab1, tab2 = st.tabs(["📊 Panel A: Análisis Exploratorio", "🔮 Panel B: Análisis Predictivo"])

with tab1:
    st.header("Análisis Exploratorio de Datos")
    
    # Filtros
    neighborhoods = st.sidebar.multiselect("Seleccionar Vecindarios", options=df['Neighborhood'].unique(), default=df['Neighborhood'].unique())
    df_filtered = df[df['Neighborhood'].isin(neighborhoods)]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros Filtrados", len(df_filtered))
    col2.metric("Precio Promedio", f"${df_filtered['SalePrice'].mean():,.2f}")
    col3.metric("Calidad Promedio", f"{df_filtered['OverallQual'].mean():.2f}/10")

    # Mapa de calor robusto
    st.subheader("Mapa de Calor de Correlaciones (Top 10 Variables)")
    corr_df = df_filtered[FEATURES + ['SalePrice']].corr()
    fig3 = px.imshow(corr_df, color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.header("🔮 Panel B: Análisis Predictivo")
    st.write("Ingrese las 10 características clave para obtener una estimación de precio.")
    
    # Formulario dinámico
    input_data = {}
    cols = st.columns(2)
    
    for i, col_name in enumerate(FEATURES):
        with cols[i % 2]:
            input_data[col_name] = [st.number_input(f"{col_name}", 
                                                  float(df[col_name].min()), 
                                                  float(df[col_name].max()), 
                                                  float(df[col_name].mean()))]

    if st.button("Calcular Predicción"):
        input_df = pd.DataFrame(input_data)
        prediction = model.predict(input_df)
        
        st.success(f"### Precio estimado: ${prediction[0]:,.2f}")
        st.info("Predicción realizada utilizando el modelo Random Forest optimizado sobre las 10 variables más relevantes.")
