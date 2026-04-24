"""
Módulo de carga y gestión de datos (Etapa 1 del pipeline).

Soporta carga desde archivos locales (CSV/TSV/JSON), desde URLs de API REST
y desde bases de datos SQL (SQLite).
Centraliza la lógica de entrada para que el resto del sistema reciba
siempre un pd.DataFrame independientemente del origen de los datos.

Clases:
    CargadorDatos: Carga datos desde archivo o URL y retorna un DataFrame.
"""

import pandas as pd
import requests
import sqlite3


class CargadorDatos:
    """
    Carga datos desde un archivo local o una URL de API.

    Parameters
    ----------
    url_base : str, optional
        URL base de la API. Por defecto cadena vacía.
    tiempo_espera : int, optional
        Tiempo límite de conexión HTTP en segundos. Por defecto 10.

    Attributes
    ----------
    url_base : str
        URL base de la API configurada.
    datos_crudos : dict
        Último conjunto de datos cargados en formato dict.

    Examples
    --------
    >>> cargador = CargadorDatos()
    >>> df = cargador.cargar_desde_url("https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY")
    >>> print(df.shape)
    """

    def __init__(self, url_base: str = "", tiempo_espera: int = 10):
        self.url_base = url_base
        self.tiempo_espera = tiempo_espera
        self.datos_crudos: dict = {}



    def cargar_desde_archivo(self, archivo) -> pd.DataFrame:
        """
        Carga datos desde un objeto archivo subido (CSV o JSON).

        Detecta la extensión del archivo y delega a _leer_csv o _leer_json.

        Parameters
        ----------
        archivo : UploadedFile
            Objeto de archivo retornado por st.file_uploader.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            Si la extensión del archivo no es .csv, .tsv ni .json.
        """
        nombre = archivo.name.lower()
        if nombre.endswith(".csv"):
            return self._leer_csv(archivo)
        elif nombre.endswith(".tsv"):
            return self._leer_tsv(archivo)
        elif nombre.endswith(".json"):
            return self._leer_json(archivo)
        else:
            raise ValueError(f"Formato no soportado: '{archivo.name}'. Use .csv, .tsv o .json.")

    def cargar_desde_ruta(self, ruta: str) -> pd.DataFrame:
        """
        Carga datos desde una ruta de archivo en disco (CSV o JSON).

        A diferencia de cargar_desde_archivo(), este método acepta una
        ruta de archivo como cadena de texto, sin necesidad de un objeto
        UploadedFile de Streamlit.

        Parameters
        ----------
        ruta : str
            Ruta al archivo CSV o JSON.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe.
        ValueError
            Si la extensión no es .csv, .tsv ni .json.
        """
        import os
        if not os.path.isfile(ruta):
            raise FileNotFoundError(f"No se encontró el archivo: '{ruta}'")

        ruta_lower = ruta.lower()
        if ruta_lower.endswith(".csv"):
            return pd.read_csv(ruta)
        elif ruta_lower.endswith(".tsv"):
            return pd.read_csv(ruta, sep='\t')
        elif ruta_lower.endswith(".json"):
            import json
            with open(ruta, "r", encoding="utf-8") as f:
                datos = json.load(f)
            self.datos_crudos = datos if isinstance(datos, dict) else {"items": datos}
            if isinstance(datos, list):
                return pd.json_normalize(datos)
            elif isinstance(datos, dict):
                for key, value in datos.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        return pd.json_normalize(value)
                return pd.json_normalize([datos])
            else:
                raise ValueError("El JSON debe ser una lista de objetos o un objeto único.")
        else:
            raise ValueError(f"Formato no soportado: '{ruta}'. Use .csv, .tsv o .json.")

    def cargar_desde_url(self, url: str) -> pd.DataFrame:
        """
        Carga datos haciendo GET a una URL y normaliza la respuesta JSON.

        Parameters
        ----------
        url : str
            URL del endpoint de API que retorna JSON.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ConnectionError
            Si no se puede conectar con la URL.
        ValueError
            Si la respuesta no es JSON válido o la estructura no es soportada.
        """
        try:
            respuesta = requests.get(url, timeout=self.tiempo_espera)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"No se pudo conectar a '{url}': {e}") from e
        except requests.exceptions.Timeout as e:
            raise ConnectionError(f"Tiempo de espera agotado al conectar a '{url}'.") from e

        self._manejar_error(respuesta)

        try:
            datos = respuesta.json()
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(f"La respuesta de '{url}' no es JSON válido.") from e

        # Almacenar datos crudos
        self.datos_crudos = datos if isinstance(datos, dict) else {"items": datos}

        if isinstance(datos, list):
            return pd.json_normalize(datos)
        elif isinstance(datos, dict):
            # Buscar listas anidadas en el diccionario (patrón común en APIs de NASA)
            for key, value in datos.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    return pd.json_normalize(value)
            return pd.json_normalize([datos])
        else:
            raise ValueError("La respuesta JSON debe ser una lista de objetos o un objeto.")

  
    def obtener_datos_api(self) -> list:
        """
        Obtiene datos desde la URL base configurada.

        Equivalente genérico de obtener_datos_lanzamientos() del UML.

        Returns
        -------
        list
            Lista de registros obtenidos de la API.
        """
        if not self.url_base:
            raise ValueError("No se ha configurado una url_base.")
        df = self.cargar_desde_url(self.url_base)
        return df.to_dict(orient="records")

    def obtener_datos_secundarios(self, url: str) -> list:
        """
        Obtiene datos desde una URL secundaria.

        Equivalente genérico de obtener_datos_cohetes() del UML.

        Parameters
        ----------
        url : str
            URL del endpoint secundario.

        Returns
        -------
        list
            Lista de registros obtenidos.
        """
        df = self.cargar_desde_url(url)
        return df.to_dict(orient="records")


    # ── Conexión a base de datos SQL ──────────────────────────────────

    def listar_tablas_sql(self, ruta_db: str) -> list:
        """
        Lista las tablas disponibles en una base de datos SQLite.

        Parameters
        ----------
        ruta_db : str
            Ruta al archivo .db / .sqlite.

        Returns
        -------
        list
            Lista de nombres de tablas.
        """
        conn = sqlite3.connect(ruta_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tablas = [fila[0] for fila in cursor.fetchall()]
        conn.close()
        return tablas

    def cargar_desde_sql(self, ruta_db: str, tabla: str) -> pd.DataFrame:
        """
        Carga una tabla completa desde una base de datos SQLite.

        Parameters
        ----------
        ruta_db : str
            Ruta al archivo .db / .sqlite.
        tabla : str
            Nombre de la tabla a cargar.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError
            Si el archivo de base de datos no existe.
        ValueError
            Si la tabla no existe en la base de datos.
        """
        import os
        if not os.path.isfile(ruta_db):
            raise FileNotFoundError(f"No se encontró la base de datos: '{ruta_db}'")

        tablas_disponibles = self.listar_tablas_sql(ruta_db)
        if tabla not in tablas_disponibles:
            raise ValueError(
                f"La tabla '{tabla}' no existe. "
                f"Tablas disponibles: {tablas_disponibles}"
            )

        conn = sqlite3.connect(ruta_db)
        df = pd.read_sql_query(f"SELECT * FROM [{tabla}]", conn)
        conn.close()
        return df

    # ── Lectores internos ────────────────────────────────────────────

    def _leer_csv(self, archivo) -> pd.DataFrame:
        """Lee un archivo CSV y retorna un DataFrame."""
        return pd.read_csv(archivo)

    def _leer_tsv(self, archivo) -> pd.DataFrame:
        """Lee un archivo TSV (tab-separated) y retorna un DataFrame."""
        return pd.read_csv(archivo, sep='\t')

    def _leer_json(self, archivo) -> pd.DataFrame:
        """Lee un archivo JSON y retorna un DataFrame normalizado."""
        import json
        datos = json.load(archivo)
        self.datos_crudos = datos if isinstance(datos, dict) else {"items": datos}
        if isinstance(datos, list):
            return pd.json_normalize(datos)
        elif isinstance(datos, dict):
            for key, value in datos.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    return pd.json_normalize(value)
            return pd.json_normalize([datos])
        else:
            raise ValueError("El JSON debe ser una lista de objetos o un objeto único.")

    def _manejar_error(self, respuesta) -> None:
        """
        Valida el código HTTP y lanza excepciones descriptivas.

        Parameters
        ----------
        respuesta : requests.Response
            Respuesta HTTP a validar.
        """
        if respuesta.status_code == 404:
            raise ConnectionError(f"Recurso no encontrado (404): {respuesta.url}")
        elif respuesta.status_code == 401:
            raise ConnectionError(f"No autorizado (401): {respuesta.url}")
        elif respuesta.status_code == 403:
            raise ConnectionError(f"Acceso prohibido (403): {respuesta.url}")
        elif respuesta.status_code >= 500:
            raise ConnectionError(
                f"Error del servidor ({respuesta.status_code}): {respuesta.url}"
            )
        elif not respuesta.ok:
            raise ConnectionError(
                f"Error HTTP {respuesta.status_code}: {respuesta.url}"
            )
