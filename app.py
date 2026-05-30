import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# -------------------------------------------------
# Configuración de la página
# -------------------------------------------------

st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="🌸",
    layout="wide"
)

# -------------------------------------------------
# Carga de datos
# -------------------------------------------------

@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)

    df = iris.frame.copy()

    species_names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    df["species"] = df["target"].map(species_names)

    return df

df = load_data()

# -------------------------------------------------
# Título
# -------------------------------------------------

st.title("🌸 Dashboard del Dataset Iris")
st.markdown(
    """
    Exploración interactiva del clásico dataset Iris utilizado
    en clasificación supervisada.
    """
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header("Filtros")

species = st.sidebar.multiselect(
    "Especies",
    options=df["species"].unique(),
    default=df["species"].unique()
)

filtered_df = df[df["species"].isin(species)]

# -------------------------------------------------
# Métricas
# -------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Observaciones", len(filtered_df))

with col2:
    st.metric(
        "Sepal Length Promedio",
        round(filtered_df["sepal length (cm)"].mean(), 2)
    )

with col3:
    st.metric(
        "Petal Length Promedio",
        round(filtered_df["petal length (cm)"].mean(), 2)
    )

with col4:
    st.metric(
        "Número de Especies",
        filtered_df["species"].nunique()
    )

st.divider()

# -------------------------------------------------
# Gráficos principales
# -------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig_scatter = px.scatter(
        filtered_df,
        x="sepal length (cm)",
        y="petal length (cm)",
        color="species",
        color_discrete_sequence=px.colors.sequential.Viridis,
        size="petal width (cm)",
        hover_data=["sepal width (cm)"],
        title="Relación entre Longitud de Sépalo y Pétalo"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:

    fig_box = px.box(
        filtered_df,
        x="species",
        y="sepal width (cm)",
        color="species",
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Distribución del Ancho del Sépalo"
    )

    st.plotly_chart(fig_box, use_container_width=True)

# -------------------------------------------------
# Histogramas
# -------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig_hist = px.histogram(
        filtered_df,
        x="petal length (cm)",
        color="species",
        barmode="overlay",
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Distribución de la Longitud del Pétalo"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

with col2:

    corr = filtered_df.drop(
        columns=["target", "species"]
    ).corr()

    fig_heatmap = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Viridis",
        title="Matriz de Correlación"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

# -------------------------------------------------
# Tabla de datos
# -------------------------------------------------

st.subheader("Datos")

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)

# -------------------------------------------------
# Resumen estadístico
# -------------------------------------------------

st.subheader("Resumen Estadístico")

st.dataframe(
    filtered_df.describe(),
    use_container_width=True
)
