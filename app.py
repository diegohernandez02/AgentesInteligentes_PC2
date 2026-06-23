import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Configuración de la página
st.set_page_config(page_title="Dashboard Ames Housing", layout="wide")

# Lista de las 10 variables (Top 10)
FEATURES = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 
            'YearBuilt', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd', 'MasVnrArea']

@st.cache_data
def load_data():
    return pd.read_csv('train.csv')

@st.cache_resource
def load_model():
    return joblib.load('modelo_final.pkl')

# Carga de datos y modelo
df = load_data()
model = load_model()

st.title("🏡 Dashboard Inmobiliario - Ames Housing")

tab1, tab2 = st.tabs(["📊 Panel A: Análisis Exploratorio", "🔮 Panel B: Análisis Predictivo"])

with tab1:
    st.header("Análisis Exploratorio de Datos")
    
    neighborhoods = st.sidebar.multiselect("Vecindarios", options=df['Neighborhood'].unique(), default=df['Neighborhood'].unique())
    df_filtered = df[df['Neighborhood'].isin(neighborhoods)]

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", len(df_filtered))
    c2.metric("Promedio Precio", f"${df_filtered['SalePrice'].mean():,.0f}")
    c3.metric("Calidad Promedio", f"{df_filtered['OverallQual'].mean():.1f}/10")

    # Gráficos
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Correlación (Top 10)")
        fig3 = px.imshow(df_filtered[FEATURES + ['SalePrice']].corr(), color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig3, use_container_width=True)
    with col_b:
        st.subheader("Área vs Precio")
        fig_scatter = px.scatter(df_filtered, x="GrLivArea", y="SalePrice", color="OverallQual", hover_data=['Neighborhood'])
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.header("🔮 Panel B: Análisis Predictivo")
    input_data = {}
    cols = st.columns(2)
    
    for i, col_name in enumerate(FEATURES):
        with cols[i % 2]:
            input_data[col_name] = st.number_input(f"{col_name}", 
                                                  float(df[col_name].min()), 
                                                  float(df[col_name].max()), 
                                                  float(df[col_name].mean()))

    if st.button("Calcular Predicción"):
        # Crear DF con una fila (orient='index' o pasando lista de valores)
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f"### Precio estimado: ${prediction[0]:,.2f}")
