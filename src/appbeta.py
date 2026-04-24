"""
MOTOR DE ANÁLISIS DE DATOS — Versión Beta

Demuestra las siguientes capacidades:
  1. Carga exitosa de archivos .csv
  2. Carga exitosa de archivos .tsv
  3. Conexión a base de datos SQL (SQLite)
  4. Visualización de tablas SQL
  5. Análisis Exploratorio de Datos (EDA) para cada fuente

Uso
----
    python src/appbeta.py
"""

import sys
import os
import io

import pandas as pd

# Forzar salida UTF-8 en Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Agregar el directorio src al path para importar módulos locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carga_datos import CargadorDatos
from analisis_eda import AnalizadorExploratorio
from preprocesamiento import Preprocesador





def imprimir_encabezado(titulo: str) -> None:
    """Imprime un encabezado visual."""
    print(f"  {titulo}")


def imprimir_seccion(titulo: str) -> None:
    """Imprime un separador de sección."""
    print(f"  {titulo}")


# Análisis Exploratorio de Datos (EDA) 

def ejecutar_eda(df: pd.DataFrame, nombre_fuente: str) -> None:
    """
    Ejecuta el análisis exploratorio completo sobre un DataFrame
    y muestra los resultados en consola.
    """
    imprimir_seccion(f"EDA — {nombre_fuente}")

    # Resumen general
    analizador = AnalizadorExploratorio(df)
    resumen = analizador.resumen_general()

    print(f"\n    Dimensiones: {resumen['total_registros']} filas × {resumen['total_columnas']} columnas")
    print(f"    Total de valores nulos: {resumen['total_nulos']}")

    print("\n    Tipos de datos por columna:")
    for col, tipo in resumen["tipos_datos"].items():
        print(f"      • {col}: {tipo}")

    # Primeras filas
    print(f"\n    Primeras 5 filas:")
    print(df.head(5).to_string(index=False))

    # Estadísticas descriptivas
    print(f"\n    Estadísticas descriptivas:")
    desc = analizador.estadisticas_descriptivas()
    print(desc.to_string())

    # Valores nulos
    nulos = analizador.detectar_nulos()
    if nulos:
        print("\n    Columnas con valores nulos (% del total):")
        for col, pct in nulos.items():
            print(f"      {col}: {pct}%")
    else:
        print("\n    Sin valores nulos en el dataset.")

    # Clasificación de columnas
    preprocesador = Preprocesador(df)
    tipos = preprocesador.detectar_tipos_columna()
    print("\n    Clasificación de columnas:")
    for categoria, columnas in tipos.items():
        if columnas:
            print(f"    {categoria.upper()}: {columnas}")

    # Detección de atípicos en columnas numéricas
    if tipos["numericas"]:
        print("\n    Detección de valores atípicos (IQR):")
        for col in tipos["numericas"]:
            atipicos = analizador.detectar_atipicos(col)
            n = len(atipicos)
            if n > 0:
                muestra = atipicos[:5]
                extra = f" ... (+{n - 5} más)" if n > 5 else ""
                print(f"      {col}: {n} atípicos → {muestra}{extra}")
            else:
                print(f"      {col}: sin atípicos")

    # Distribución de categorías
    if tipos["categoricas"]:
        print("\n    Distribución de columnas categóricas:")
        for col in tipos["categoricas"]:
            dist = analizador.distribucion_por_categoria(col)
            print(f"       {col}:")
            for cat, freq in list(dist.items())[:10]:
                print(f"          {cat}: {freq}")
            if len(dist) > 10:
                print(f"          ... y {len(dist) - 10} categorías más")

    print(f"\n    EDA completado para '{nombre_fuente}'.")


# Funciones de carga 

def cargar_csv(cargador: CargadorDatos) -> tuple:
    """Carga un archivo CSV y retorna (DataFrame, nombre)."""
    imprimir_encabezado("CARGA DE ARCHIVO CSV")
    ruta = input("  Ingresa la ruta del archivo .csv: ").strip()
    if not ruta:
        print("  No se proporcionó ninguna ruta.")
        return None, None
    df = cargador.cargar_desde_ruta(ruta)
    nombre = os.path.basename(ruta)
    print(f"  Archivo '{nombre}' cargado exitosamente.")
    print(f"    Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"\n    Primeras 3 filas:")
    print(df.head(3).to_string(index=False))
    return df, nombre


def cargar_tsv(cargador: CargadorDatos) -> tuple:
    """Carga un archivo TSV y retorna (DataFrame, nombre)."""
    imprimir_encabezado("CARGA DE ARCHIVO TSV")
    ruta = input("  Ingresa la ruta del archivo .tsv: ").strip()
    if not ruta:
        print("  No se proporcionó ninguna ruta.")
        return None, None
    df = cargador.cargar_desde_ruta(ruta)
    nombre = os.path.basename(ruta)
    print(f"  Archivo '{nombre}' cargado exitosamente.")
    print(f"    Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"\n    Primeras 3 filas:")
    print(df.head(3).to_string(index=False))
    return df, nombre


def conectar_sql(cargador: CargadorDatos) -> tuple:
    """Conecta a una base de datos SQLite y carga una tabla."""
    imprimir_encabezado("CONEXIÓN A BASE DE DATOS SQL (SQLite)")
    ruta_db = input("  Ingresa la ruta de la base de datos (.db/.sqlite): ").strip()
    if not ruta_db:
        print("  No se proporcionó ninguna ruta.")
        return None, None

    # Listar tablas disponibles
    tablas = cargador.listar_tablas_sql(ruta_db)
    if not tablas:
        print("  La base de datos no contiene tablas.")
        return None, None

    print(f"\n  Conexión exitosa a '{os.path.basename(ruta_db)}'")
    print(f"    Tablas disponibles ({len(tablas)}):")
    for i, tabla in enumerate(tablas, 1):
        print(f"      {i}. {tabla}")

    seleccion = input("\n  Selecciona el número de la tabla a cargar: ").strip()
    try:
        idx = int(seleccion) - 1
        if idx < 0 or idx >= len(tablas):
            print(" Selección fuera de rango.")
            return None, None
        tabla = tablas[idx]
    except ValueError:
        print("  Entrada no válida.")
        return None, None

    df = cargador.cargar_desde_sql(ruta_db, tabla)
    nombre = f"SQL:{os.path.basename(ruta_db)}.{tabla}"
    print(f"\n  ✓ Tabla '{tabla}' cargada exitosamente.")
    print(f"    Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"\n    Primeras 3 filas:")
    print(df.head(3).to_string(index=False))
    return df, nombre


def visualizar_tablas_sql(cargador: CargadorDatos) -> None:
    """Muestra todas las tablas y sus esquemas de una base de datos SQL."""
    imprimir_encabezado("VISUALIZACIÓN DE TABLAS SQL")
    ruta_db = input("  Ingresa la ruta de la base de datos (.db/.sqlite): ").strip()
    if not ruta_db:
        print("  No se proporcionó ninguna ruta.")
        return

    import sqlite3

    if not os.path.isfile(ruta_db):
        print(f"  No se encontró la base de datos '{ruta_db}'.")
        return

    tablas = cargador.listar_tablas_sql(ruta_db)
    if not tablas:
        print("  La base de datos no contiene tablas.")
        return

    print(f"\n  Base de datos: '{os.path.basename(ruta_db)}'")
    print(f"    Total de tablas: {len(tablas)}")

    conn = sqlite3.connect(ruta_db)

    for tabla in tablas:
        print(f"\n  ┌─ Tabla: {tabla}")

        # Obtener esquema de la tabla
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info([{tabla}])")
        columnas_info = cursor.fetchall()

        print(f"  │  Columnas ({len(columnas_info)}):")
        for col_info in columnas_info:
            pk = " [PK]" if col_info[5] else ""
            nullable = "" if col_info[3] else " (nullable)"
            print(f"  │    • {col_info[1]} ({col_info[2]}){pk}{nullable}")

        # Contar registros
        cursor.execute(f"SELECT COUNT(*) FROM [{tabla}]")
        total = cursor.fetchone()[0]
        print(f"  │  Total de registros: {total}")

        # Mostrar primeras filas
        df_preview = pd.read_sql_query(f"SELECT * FROM [{tabla}] LIMIT 5", conn)
        print(f"  │  Primeras 5 filas:")
        for linea in df_preview.to_string(index=False).split("\n"):
            print(f"  │    {linea}")

        print(f"  └{'─' * 50}")

    conn.close()
    print(f"\n  Visualización completada.")


# ── Menú principal ────────────────────────────────────────────────────

def mostrar_menu() -> None:
    """Muestra el menú principal."""
    print("\n  Fuentes de datos disponibles:")
    print("    1. Cargar archivo CSV")
    print("    2. Cargar archivo TSV")
    print("    3. Conectar a base de datos SQL")
    print("    4. Visualizar tablas SQL")
    print()
    print("  Análisis:")
    print("    5. Ejecutar EDA sobre fuentes cargadas")
    print()
    print("    0. Salir")
    print()


def ejecutar_pipeline():
    """Pipeline principal con menú interactivo."""
    cargador = CargadorDatos()

    # Diccionario para almacenar múltiples fuentes cargadas
    fuentes_cargadas: dict = {}  # {nombre: DataFrame}

    while True:
        mostrar_menu()

        if fuentes_cargadas:
            print(f"  Fuentes cargadas ({len(fuentes_cargadas)}):")
            for i, nombre in enumerate(fuentes_cargadas, 1):
                df = fuentes_cargadas[nombre]
                print(f"    {i}. {nombre} ({df.shape[0]}×{df.shape[1]})")
            print()

        opcion = input("  Selecciona una opción: ").strip()

        try:
            if opcion == "1":
                df, nombre = cargar_csv(cargador)
                if df is not None:
                    fuentes_cargadas[nombre] = df

            elif opcion == "2":
                df, nombre = cargar_tsv(cargador)
                if df is not None:
                    fuentes_cargadas[nombre] = df

            elif opcion == "3":
                df, nombre = conectar_sql(cargador)
                if df is not None:
                    fuentes_cargadas[nombre] = df

            elif opcion == "4":
                visualizar_tablas_sql(cargador)

            elif opcion == "5":
                if not fuentes_cargadas:
                    print("\n  No hay fuentes cargadas. Carga al menos un archivo o tabla primero.")
                else:
                    imprimir_encabezado("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
                    print(f"\n  Se ejecutará EDA sobre {len(fuentes_cargadas)} fuente(s):\n")
                    for nombre in fuentes_cargadas:
                        print(f"    • {nombre}")

                    for nombre, df in fuentes_cargadas.items():
                        ejecutar_eda(df, nombre)

                    print(f"  EDA completado para todas las fuentes.")

            elif opcion == "0":
                
                break

            else:
                print("\n  Opción no válida. Intenta de nuevo.")

        except FileNotFoundError as e:
            print(f"\n Error: {e}")
        except ValueError as e:
            print(f"\n   Error: {e}")
        except Exception as e:
            print(f"\n  Error inesperado: {e}")

        input("\n  Presiona Enter para continuar...")


if __name__ == "__main__":
    ejecutar_pipeline()