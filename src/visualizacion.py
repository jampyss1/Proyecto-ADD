"""
Módulo de visualización de resultados (Etapa 6 del pipeline).

Genera gráficas interactivas con Plotly que se renderizan
en el dashboard Streamlit.

Clases:
    Visualizador: Genera figuras Plotly listas para renderizar.
"""

import plotly.express as px
import plotly.graph_objects as go


class Visualizador:
    """
    Genera gráficas interactivas con Plotly.

    Todos los métodos son funciones puras: reciben datos y devuelven
    una figura, sin almacenar estado interno.
    """

    def graficar_linea_temporal(self, datos: dict) -> go.Figure:
        """
        Genera gráfica de línea con evolución temporal.

        Parameters
        ----------
        datos : dict
            {periodo: valor_promedio} ordenado temporalmente.

        Returns
        -------
        go.Figure
        """
        if not datos:
            return None
        etiquetas = list(datos.keys())
        valores = list(datos.values())
        fig = px.line(
            x=etiquetas,
            y=valores,
            markers=True,
            labels={"x": "Periodo", "y": "Valor"},
            title="Tendencia temporal",
        )
        fig.update_layout(template="plotly_dark")
        return fig

    def graficar_barras_comparativas(self, datos: dict) -> go.Figure:
        """
        Genera gráfica de barras para comparar entidades.

        Parameters
        ----------
        datos : dict
            {categoría: conteo}.

        Returns
        -------
        go.Figure
        """
        if not datos:
            return None
        categorias = list(datos.keys())
        conteos = list(datos.values())
        fig = px.bar(
            x=categorias,
            y=conteos,
            labels={"x": "Categoría", "y": "Conteo"},
            title="Comparación por categoría",
            color=conteos,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        return fig

    def graficar_frecuencia_mensual(self, datos: dict) -> go.Figure:
        """
        Muestra distribución de registros por mes del año.

        Parameters
        ----------
        datos : dict
            {mes: frecuencia}.

        Returns
        -------
        go.Figure
        """
        if not datos:
            return None
        meses = list(datos.keys())
        frecuencias = list(datos.values())
        fig = px.bar(
            x=meses,
            y=frecuencias,
            labels={"x": "Mes", "y": "Frecuencia"},
            title="Frecuencia mensual",
            color=frecuencias,
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        return fig

    def graficar_comparacion_numerica(self, datos: dict) -> go.Figure:
        """
        Compara una métrica numérica entre categorías con barras agrupadas.

        Parameters
        ----------
        datos : dict
            {categoría: valor_numérico}.

        Returns
        -------
        go.Figure
        """
        if not datos:
            return None
        categorias = list(datos.keys())
        valores = list(datos.values())
        fig = px.bar(
            x=categorias,
            y=valores,
            labels={"x": "Categoría", "y": "Valor"},
            title="Comparación numérica",
            color=valores,
            color_continuous_scale="Plasma",
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        return fig

    def graficar_metricas_evaluacion(self, datos: dict) -> go.Figure:
        """
        Visualiza las métricas de evaluación del modelo.

        Parameters
        ----------
        datos : dict
            {nombre_metrica: valor} — ej. {'R²': 0.85, 'RMSE': 1.2, 'MAE': 0.9}

        Returns
        -------
        go.Figure
        """
        if not datos:
            return None
        metricas = list(datos.keys())
        valores = list(datos.values())
        fig = go.Figure(
            data=[
                go.Bar(
                    x=metricas,
                    y=valores,
                    marker_color=["#636EFA", "#EF553B", "#00CC96"],
                    text=[f"{v:.4f}" for v in valores],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title="Métricas de evaluación del modelo",
            xaxis_title="Métrica",
            yaxis_title="Valor",
            template="plotly_dark",
        )
        return fig
