"""
Módulo de preprocesamiento (Etapa 3 del pipeline).

Filtra, limpia y transforma cualquier DataFrame para que pueda ser
consumido por los modelos de análisis.

Clases:
    Preprocesador: Aplica filtros genéricos, maneja nulos y normaliza datos.
"""

import pandas as pd


class Preprocesador:
    """
    Filtra, limpia y transforma un DataFrame según los criterios del usuario.

    Parameters
    ----------
    datos : pd.DataFrame or list
        DataFrame o lista de diccionarios con los datos crudos a procesar.

    Attributes
    ----------
    datos_brutos : list
        Datos originales en formato lista de diccionarios (alineado con UML).
    """

    def __init__(self, datos):
        if isinstance(datos, list):
            self._datos_brutos = datos
            self._datos = pd.DataFrame(datos)
        elif isinstance(datos, pd.DataFrame):
            self._datos_brutos = datos.to_dict(orient="records")
            self._datos = datos
        else:
            raise TypeError("Se espera un pd.DataFrame o una lista de diccionarios.")

    @property
    def datos_brutos(self) -> list:
        """Acceso a los datos brutos en formato lista."""
        return self._datos_brutos

  
    # Detección de tipos de columna
 
    def detectar_tipos_columna(self) -> dict:
        """
        Clasifica las columnas del DataFrame por tipo de dato.

        Returns
        -------
        dict
            Diccionario con las claves 'numericas', 'categoricas',
            'fechas' y 'texto_libre', cada una con la lista de
            nombres de columnas correspondientes.
        """
        numericas = []
        categoricas = []
        fechas = []
        texto_libre = []

        for col in self._datos.columns:
            dtype = self._datos[col].dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                fechas.append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                numericas.append(col)
            elif pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                n_unicos = self._datos[col].nunique(dropna=True)
                n_total = len(self._datos)
                # Heurística: si hay muchos valores únicos respecto al total → texto libre
                if n_unicos > min(50, n_total * 0.5):
                    texto_libre.append(col)
                else:
                    categoricas.append(col)

        return {
            "numericas": numericas,
            "categoricas": categoricas,
            "fechas": fechas,
            "texto_libre": texto_libre,
        }

    # Filtros genéricos
  

    def filtrar_por_categoria(self, columna: str, valores: list) -> pd.DataFrame:
        """
        Filtra filas cuya columna contenga alguno de los valores indicados.

        Parameters
        ----------
        columna : str
            Nombre de la columna categórica.
        valores : list
            Lista de valores a incluir.

        Returns
        -------
        pd.DataFrame
        """
        if not valores:
            return self._datos.copy()
        return self._datos[self._datos[columna].isin(valores)].copy()

    def filtrar_por_rango_numerico(self, columna: str, minimo: float, maximo: float) -> pd.DataFrame:
        """
        Filtra filas cuyo valor numérico esté dentro del rango [minimo, maximo].

        Parameters
        ----------
        columna : str
            Nombre de la columna numérica.
        minimo : float
            Valor mínimo (inclusive).
        maximo : float
            Valor máximo (inclusive).

        Returns
        -------
        pd.DataFrame
        """
        return self._datos[
            (self._datos[columna] >= minimo) & (self._datos[columna] <= maximo)
        ].copy()

    def filtrar_por_anio(self, anio: int, columna_fecha: str = None) -> list:
        """
        Filtra registros por año en una columna de fecha.

        Equivalente genérico de filtrar_por_anio del UML.

        Parameters
        ----------
        anio : int
            Año a filtrar.
        columna_fecha : str, optional
            Nombre de la columna de fecha. Si no se especifica,
            se busca la primera columna tipo datetime.

        Returns
        -------
        list
            Lista de registros que coinciden con el año.
        """
        if columna_fecha is None:
            cols_fecha = [
                col for col in self._datos.columns
                if pd.api.types.is_datetime64_any_dtype(self._datos[col])
            ]
            if not cols_fecha:
                return self._datos_brutos
            columna_fecha = cols_fecha[0]

        mascara = self._datos[columna_fecha].dt.year == anio
        return self._datos[mascara].to_dict(orient="records")

    def filtrar_por_exito(self, exitoso: bool, columna_exito: str = None) -> list:
        """
        Filtra registros por una columna booleana de éxito/fracaso.

        Equivalente genérico de filtrar_por_exito del UML.

        Parameters
        ----------
        exitoso : bool
            True para registros exitosos, False para no exitosos.
        columna_exito : str, optional
            Nombre de la columna booleana. Si no se especifica,
            se busca la primera columna de tipo bool.

        Returns
        -------
        list
            Lista de registros filtrados.
        """
        if columna_exito is None:
            cols_bool = [
                col for col in self._datos.columns
                if pd.api.types.is_bool_dtype(self._datos[col])
            ]
            if not cols_bool:
                return self._datos_brutos
            columna_exito = cols_bool[0]

        mascara = self._datos[columna_exito] == exitoso
        return self._datos[mascara].to_dict(orient="records")

    # Limpieza y transformación

    def limpiar_nulos(self, datos: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja valores nulos: elimina filas con todos nulos
        y rellena restantes con media (numéricas) o moda (categóricas).

        Parameters
        ----------
        datos : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        resultado = datos.dropna(how="all").copy()
        for col in resultado.columns:
            if pd.api.types.is_numeric_dtype(resultado[col]):
                resultado[col] = resultado[col].fillna(resultado[col].mean())
            else:
                moda = resultado[col].mode()
                if not moda.empty:
                    resultado[col] = resultado[col].fillna(moda.iloc[0])
        return resultado

    def normalizar_fechas(self, datos: pd.DataFrame, columnas_fecha: list = None) -> pd.DataFrame:
        """
        Convierte las columnas indicadas a formato datetime estándar.

        Parameters
        ----------
        datos : pd.DataFrame
        columnas_fecha : list, optional
            Lista de nombres de columna. Si es None, se autodetectan
            columnas tipo objeto con formato de fecha.

        Returns
        -------
        pd.DataFrame
        """
        resultado = datos.copy()

        if columnas_fecha is None:
            columnas_fecha = []
            for col in resultado.columns:
                if pd.api.types.is_object_dtype(resultado[col]):
                    muestra = resultado[col].dropna().head(5)
                    try:
                        pd.to_datetime(muestra)
                        columnas_fecha.append(col)
                    except (ValueError, TypeError):
                        pass

        for col in columnas_fecha:
            if col in resultado.columns:
                resultado[col] = pd.to_datetime(resultado[col], errors="coerce", utc=True)
        return resultado

    def a_dataframe(self) -> pd.DataFrame:
        """
        Convierte los datos brutos (lista) a un DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(self._datos_brutos)

    def obtener_anios_disponibles(self, columna_fecha: str = None) -> list:
        """
        Retorna los años únicos presentes en una columna de fecha.

        Parameters
        ----------
        columna_fecha : str, optional
            Columna de fecha. Si no se especifica, se autodetecta.

        Returns
        -------
        list
            Lista de años disponibles, ordenada.
        """
        if columna_fecha is None:
            cols_fecha = [
                col for col in self._datos.columns
                if pd.api.types.is_datetime64_any_dtype(self._datos[col])
            ]
            if not cols_fecha:
                return []
            columna_fecha = cols_fecha[0]

        anios = self._datos[columna_fecha].dropna().dt.year.unique().tolist()
        return sorted(anios)

    # Utilidades para filtros dinámicos del sidebar

    def obtener_valores_unicos(self, columna: str) -> list:
        """
        Retorna los valores únicos (no nulos) de una columna, ordenados.

        Parameters
        ----------
        columna : str

        Returns
        -------
        list
        """
        return sorted(self._datos[columna].dropna().unique().tolist())

    def obtener_rango(self, columna: str) -> tuple:
        """
        Retorna el valor mínimo y máximo de una columna numérica.

        Parameters
        ----------
        columna : str

        Returns
        -------
        tuple
            (min, max) de la columna.
        """
        return (float(self._datos[columna].min()), float(self._datos[columna].max()))
