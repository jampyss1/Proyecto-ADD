"""
Módulo de modelado (Etapa 4 del pipeline).

Aplica técnicas estadísticas y de machine learning para extraer
patrones, tendencias y predicciones de cualquier DataFrame.

Clases:
    Modelador: Implementa modelos predictivos genéricos sobre los datos.
"""

import numpy as np
import pandas as pd


class Modelador:
    """
    Aplica modelos estadísticos y de ML sobre los datos preprocesados.

    Parameters
    ----------
    datos : pd.DataFrame
        DataFrame con los datos ya filtrados y limpios.

    Attributes
    ----------
    datos : pd.DataFrame
        Datos de trabajo.
    modelo : object or None
        Modelo de regresión entrenado.
    predicciones : pd.Series or None
        Predicciones generadas por el último modelo entrenado.
    """

    def __init__(self, datos: pd.DataFrame):
        self._datos = datos
        self._modelo = None
        self._predicciones = None
        self._col_x = None
        self._col_y = None


    def conteo_por_categoria(self, columna: str) -> dict:
        """
        Cuenta la frecuencia de cada valor en una columna categórica.

        Equivalente genérico de lanzamientos_por_cohete del UML.

        Parameters
        ----------
        columna : str

        Returns
        -------
        dict
            {categoria: conteo} ordenado de mayor a menor.
        """
        conteos = self._datos[columna].value_counts()
        return conteos.to_dict()

    def tasa_exito_por_anio(self, columna_fecha: str = None, columna_exito: str = None) -> dict:
        """
        Calcula la tasa de éxito (proporción de True) agrupada por año.

        Equivalente genérico de tasa_exito_por_anio del UML.

        Parameters
        ----------
        columna_fecha : str, optional
            Columna de fecha. Se autodetecta si no se especifica.
        columna_exito : str, optional
            Columna booleana de éxito. Se autodetecta si no se especifica.

        Returns
        -------
        dict
            {año: tasa_exito} ordenado cronológicamente.
        """
        df = self._datos.copy()

        if columna_fecha is None:
            cols_fecha = [
                c for c in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[c])
            ]
            if not cols_fecha:
                return {}
            columna_fecha = cols_fecha[0]

        if columna_exito is None:
            cols_bool = [
                c for c in df.columns
                if pd.api.types.is_bool_dtype(df[c])
            ]
            if not cols_bool:
                return {}
            columna_exito = cols_bool[0]

        df["_anio"] = df[columna_fecha].dt.year
        tasa = df.groupby("_anio")[columna_exito].mean()
        return tasa.to_dict()

    def promedio_por_categoria(self, columna_grupo: str, columna_valor: str) -> dict:
        """
        Calcula el promedio de una columna numérica agrupado por categoría.

        Parameters
        ----------
        columna_grupo : str
        columna_valor : str

        Returns
        -------
        dict
            {categoria: promedio}.
        """
        resultado = self._datos.groupby(columna_grupo)[columna_valor].mean()
        return resultado.to_dict()


    def analisis_tendencia(self, columna_temporal: str, columna_valor: str) -> dict:
        """
        Detecta si la tendencia de una serie temporal es creciente,
        estable o decreciente usando la pendiente de regresión lineal.

        Parameters
        ----------
        columna_temporal : str
        columna_valor : str

        Returns
        -------
        dict
            {'tendencia': str, 'pendiente': float, 'serie': dict}
        """
        from sklearn.linear_model import LinearRegression

        df = self._datos[[columna_temporal, columna_valor]].dropna()
        if pd.api.types.is_datetime64_any_dtype(df[columna_temporal]):
            df = df.copy()
            df[columna_temporal] = df[columna_temporal].dt.year

        serie = df.groupby(columna_temporal)[columna_valor].mean()
        x = serie.index.values.reshape(-1, 1).astype(float)
        y = serie.values.astype(float)

        modelo = LinearRegression().fit(x, y)
        pendiente = float(modelo.coef_[0])

        if pendiente > 0.01:
            tendencia = "creciente"
        elif pendiente < -0.01:
            tendencia = "decreciente"
        else:
            tendencia = "estable"

        return {
            "tendencia": tendencia,
            "pendiente": pendiente,
            "serie": serie.to_dict(),
        }


    def entrenar_regresion_lineal(self, columna_x: str, columna_y: str) -> None:
        """
        Entrena un modelo de regresión lineal entre dos columnas.

        Parameters
        ----------
        columna_x : str
        columna_y : str

        Raises
        ------
        ValueError
            Si hay menos de 3 filas de datos válidos.
        """
        from sklearn.linear_model import LinearRegression

        df = self._datos[[columna_x, columna_y]].dropna()
        if len(df) < 3:
            raise ValueError(
                f"Se necesitan al menos 3 filas válidas para entrenar el modelo "
                f"(se encontraron {len(df)})."
            )

        x = df[columna_x].values.reshape(-1, 1).astype(float)
        y = df[columna_y].values.astype(float)

        self._modelo = LinearRegression().fit(x, y)
        self._predicciones = pd.Series(self._modelo.predict(x), name="predicciones")
        self._col_x = columna_x
        self._col_y = columna_y

    def predecir(self, valor_x: float) -> float:
        """
        Predice el valor de y para un x dado usando el modelo entrenado.

        Parameters
        ----------
        valor_x : float

        Returns
        -------
        float

        Raises
        ------
        RuntimeError
            Si el modelo no ha sido entrenado.
        """
        if self._modelo is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llame a entrenar_regresion_lineal primero.")
        return float(self._modelo.predict([[valor_x]])[0])

    def predecir_proximo_anio(self, columna_temporal: str = None, columna_valor: str = None) -> int:
        """
        Predice el valor numérico para el próximo año/periodo temporal.

        Equivalente genérico de predecir_proximo_anio del UML.

        Parameters
        ----------
        columna_temporal : str, optional
        columna_valor : str, optional

        Returns
        -------
        int
            Predicción redondeada para el siguiente periodo.
        """
        if self._modelo is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llame a entrenar_regresion_lineal primero.")

        # Usar los datos del modelo entrenado para encontrar el máximo periodo
        df = self._datos[[self._col_x]].dropna()
        max_x = float(df[self._col_x].max())
        proximo = max_x + 1
        return int(round(self.predecir(proximo)))

    def obtener_coeficientes(self) -> dict:
        """
        Retorna los coeficientes del modelo entrenado.

        Returns
        -------
        dict
            {'pendiente': float, 'intercepto': float}

        Raises
        ------
        RuntimeError
            Si el modelo no ha sido entrenado.
        """
        if self._modelo is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llame a entrenar_regresion_lineal primero.")
        return {
            "pendiente": float(self._modelo.coef_[0]),
            "intercepto": float(self._modelo.intercept_),
        }

    def obtener_predicciones(self) -> pd.Series:
        """
        Retorna las predicciones del último modelo entrenado.

        Returns
        -------
        pd.Series or None
        """
        return self._predicciones

    def obtener_valores_reales(self) -> pd.Series:
        """
        Retorna los valores reales usados en el entrenamiento.

        Returns
        -------
        pd.Series or None
        """
        if self._col_y is None:
            return None
        df = self._datos[[self._col_x, self._col_y]].dropna()
        return df[self._col_y].reset_index(drop=True)
