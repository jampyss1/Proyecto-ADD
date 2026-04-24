"""
Módulo de análisis exploratorio de datos — EDA (Etapa 2 del pipeline).

Genera estadísticas descriptivas, detecta valores nulos o atípicos,
y produce resúmenes que orientan las decisiones de preprocesamiento.

Clases:
    AnalizadorExploratorio: Realiza el análisis exploratorio de los datos.
"""

import pandas as pd
import numpy as np


class AnalizadorExploratorio:
    """
    Realiza el análisis exploratorio de datos (EDA).

    Comprende la estructura, distribución y características generales
    de los datos antes de procesarlos.

    Parameters
    ----------
    datos : pd.DataFrame
        DataFrame con los datos crudos a explorar.
    """

    def __init__(self, datos: pd.DataFrame):
        self._datos = datos

    def resumen_general(self) -> dict:
        """
        Genera un resumen general del dataset.

        Returns
        -------
        dict
            Contiene total de registros, columnas, tipos de datos y nulos.
        """
        return {
            "total_registros": len(self._datos),
            "total_columnas": len(self._datos.columns),
            "tipos_datos": self._datos.dtypes.astype(str).to_dict(),
            "total_nulos": int(self._datos.isnull().sum().sum()),
        }

    def estadisticas_descriptivas(self) -> pd.DataFrame:
        """
        Calcula estadísticas descriptivas (media, mediana, std, etc.).

        Returns
        -------
        pd.DataFrame
            Tabla con las estadísticas por columna numérica.
        """
        return self._datos.describe(include="all")

    def detectar_nulos(self) -> dict:
        """
        Identifica columnas con valores nulos y su porcentaje.

        Returns
        -------
        dict
            {columna: porcentaje_nulos} para columnas con nulos > 0.
        """
        total = len(self._datos)
        if total == 0:
            return {}
        nulos = self._datos.isnull().sum()
        nulos_pct = (nulos / total * 100).round(2)
        return {col: pct for col, pct in nulos_pct.items() if pct > 0}

    def detectar_atipicos(self, columna: str) -> list:
        """
        Detecta valores atípicos en una columna usando IQR.

        Parameters
        ----------
        columna : str
            Nombre de la columna a analizar.

        Returns
        -------
        list
            Valores identificados como atípicos.
        """
        serie = self._datos[columna].dropna()
        if not pd.api.types.is_numeric_dtype(serie):
            return []
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        atipicos = serie[(serie < limite_inferior) | (serie > limite_superior)]
        return atipicos.tolist()

    def distribucion_por_categoria(self, columna: str) -> dict:
        """
        Cuenta la frecuencia de cada categoría en una columna.

        Parameters
        ----------
        columna : str
            Nombre de la columna categórica.

        Returns
        -------
        dict
            {categoría: frecuencia}.
        """
        return self._datos[columna].value_counts().to_dict()
