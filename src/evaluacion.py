"""
Módulo de evaluación (Etapa 5 del pipeline).

Mide la calidad y confiabilidad de los resultados del modelado
mediante métricas objetivas.

Clases:
    Evaluador: Calcula métricas de desempeño del modelo.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Evaluador:
    """
    Evalúa el desempeño de los modelos mediante métricas estándar.

    Parameters
    ----------
    valores_reales : pd.Series
        Valores observados reales.
    predicciones : pd.Series
        Valores predichos por el modelo.
    """

    def __init__(self, valores_reales: pd.Series, predicciones: pd.Series):
        self._valores_reales = valores_reales
        self._predicciones = predicciones

    def calcular_r2(self) -> float:
        """Calcula el coeficiente de determinación R²."""
        return float(r2_score(self._valores_reales, self._predicciones))

    def calcular_rmse(self) -> float:
        """Calcula la raíz del error cuadrático medio (RMSE)."""
        mse = mean_squared_error(self._valores_reales, self._predicciones)
        return float(np.sqrt(mse))

    def calcular_mae(self) -> float:
        """Calcula el error absoluto medio (MAE)."""
        return float(mean_absolute_error(self._valores_reales, self._predicciones))

    def generar_reporte(self) -> dict:
        """
        Genera un reporte completo de evaluación.

        Returns
        -------
        dict
            Contiene R², RMSE, MAE y diagnóstico de calidad.
        """
        r2 = self.calcular_r2()
        rmse = self.calcular_rmse()
        mae = self.calcular_mae()

        if r2 >= 0.8:
            diagnostico = "Excelente — el modelo explica la mayoría de la variabilidad."
        elif r2 >= 0.5:
            diagnostico = "Aceptable — el modelo captura una parte significativa."
        elif r2 >= 0.0:
            diagnostico = "Débil — el modelo explica poca variabilidad."
        else:
            diagnostico = "Muy pobre — el modelo es peor que la media simple."

        return {
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "diagnostico": diagnostico,
        }
