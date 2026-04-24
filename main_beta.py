"""
main_beta.py — Versión beta de consola.

Ejecuta el pipeline completo de ciencia de datos sobre cualquier
archivo CSV sin necesidad de interfaz gráfica.

Uso:
    python main_beta.py ruta/al/archivo.csv
    python main_beta.py                       (pide la ruta interactivamente)
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.carga_datos import CargadorDatos
from src.analisis_eda import AnalizadorExploratorio
from src.preprocesamiento import Preprocesador
from src.modelado import Modelador
from src.evaluacion import Evaluador






def imprimir_dict(d: dict, indent: int = 2) -> None:
    """Imprime un diccionario con formato legible."""
    for clave, valor in d.items():
        print(f"{' ' * indent}{clave}: {valor}")


def imprimir_tabla(df, max_filas: int = 10) -> None:
    """Imprime un DataFrame como tabla de texto."""
    print(df.head(max_filas).to_string())
    if len(df) > max_filas:
        print(f"  ... ({len(df)} filas en total, mostrando {max_filas})")



def ejecutar_pipeline(ruta_csv: str) -> None:
    """
    Ejecuta el pipeline completo:
    Carga → Preprocesamiento → EDA → Modelado → Evaluación.
    """

    # 1. CARGA DE DATOS 
    print("1. CARGA DE DATOS")
    cargador = CargadorDatos()
    df = cargador.cargar_desde_ruta(ruta_csv)
    print(f"  Archivo cargado: {ruta_csv}")
    print(f"  Filas: {len(df)}  |  Columnas: {len(df.columns)}")
    print(f"  Columnas: {', '.join(df.columns.tolist())}")

    print("Vista previa de los datos")
    imprimir_tabla(df, max_filas=5)

    #2. PREPROCESAMIENTO
    print("\n2. PREPROCESAMIENTO")
    preprocesador = Preprocesador(df)

    # Detectar tipos
    tipos = preprocesador.detectar_tipos_columna()
    print("  Tipos de columna detectados:")
    for tipo, cols in tipos.items():
        if cols:
            print(f"    {tipo}: {', '.join(cols)}")

    # Normalizar fechas
    df = preprocesador.normalizar_fechas(df)
    print("Fechas normalizadas")

    # Limpiar nulos
    df_limpio = preprocesador.limpiar_nulos(df)
    nulos_antes = int(df.isnull().sum().sum())
    nulos_despues = int(df_limpio.isnull().sum().sum())
    print(f"Nulos limpiados: {nulos_antes} → {nulos_despues}")

    #3. ANÁLISIS EXPLORATORIO 
    print("\n3. ANÁLISIS EXPLORATORIO (EDA)")
    analizador = AnalizadorExploratorio(df_limpio)

    # Resumen general
    resumen = analizador.resumen_general()
    print("  Resumen general:")
    print(f"    Total registros: {resumen['total_registros']}")
    print(f"    Total columnas:  {resumen['total_columnas']}")
    print(f"    Total nulos:     {resumen['total_nulos']}")

    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas")
    stats = analizador.estadisticas_descriptivas()
    if stats is not None:
        imprimir_tabla(stats)

    # Valores nulos
    nulos = analizador.detectar_nulos()
    if nulos:
        print("\n  Columnas con valores nulos:")
        imprimir_dict(nulos, indent=4)
    else:
        print("\n  No se detectaron valores nulos.")

    # Valores atípicos (primera columna numérica)
    numericas = tipos.get("numericas", [])
    if numericas:
        col_atipicos = numericas[0]
        atipicos = analizador.detectar_atipicos(col_atipicos)
        if atipicos:
            print(f"\n  Valores atípicos en '{col_atipicos}': {len(atipicos)} encontrados")
            print(f"    Primeros 10: {atipicos[:10]}")
        else:
            print(f"\n  Sin valores atípicos en '{col_atipicos}'.")

    # Distribución categórica (primera columna categórica)
    categoricas = tipos.get("categoricas", [])
    if categoricas:
        col_cat = categoricas[0]
        distribucion = analizador.distribucion_por_categoria(col_cat)
        print(f"\n  Distribución de '{col_cat}':")
        imprimir_dict(distribucion, indent=4)

    #4. MODELADO 
    print("\n4. MODELADO")
    if len(numericas) >= 2:
        col_x = numericas[0]
        col_y = numericas[1]
        print(f"  Regresión lineal: {col_x} → {col_y}")

        modelador = Modelador(df_limpio)
        try:
            modelador.entrenar_regresion_lineal(col_x, col_y)
            coefs = modelador.obtener_coeficientes()
            
            print(f"    Pendiente:   {coefs['pendiente']:.4f}")
            print(f"    Intercepto:  {coefs['intercepto']:.4f}")

            # Predicción de ejemplo
            import pandas as pd
            valor_ejemplo = float(df_limpio[col_x].mean())
            prediccion = modelador.predecir(valor_ejemplo)
            print(f"\n  Predicción de ejemplo:")
            print(f"    Si {col_x} = {valor_ejemplo:.2f}  →  {col_y} ≈ {prediccion:.4f}")

            #5. EVALUACIÓN
            print("\n5. EVALUACIÓN DEL MODELO")
            predicciones = modelador.obtener_predicciones()
            valores_reales = modelador.obtener_valores_reales()

            if predicciones is not None and valores_reales is not None:
                evaluador = Evaluador(valores_reales, predicciones)
                reporte = evaluador.generar_reporte()

                print(f"  R²:          {reporte['r2']:.4f}")
                print(f"  RMSE:        {reporte['rmse']:.4f}")
                print(f"  MAE:         {reporte['mae']:.4f}")
                print(f"  Diagnóstico: {reporte['diagnostico']}")
            else:
                print("  No se pudieron obtener predicciones para evaluar.")

        except Exception as e:
            print(f"  Error al entrenar el modelo: {e}")
    else:
        print("  Se necesitan al menos 2 columnas numéricas para el modelado.")
        print("    Columnas numéricas encontradas:", numericas)

    # Conteo por categoría 
    if categoricas and numericas:
        print("Promedio numérico por categoría")
        col_grupo = categoricas[0]
        col_valor = numericas[0]
        modelador_extra = Modelador(df_limpio)
        promedios = modelador_extra.promedio_por_categoria(col_grupo, col_valor)
        print(f"  Promedio de '{col_valor}' por '{col_grupo}':")
        imprimir_dict(promedios, indent=4)

    


# main

def main():
    """Punto de entrada principal."""
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        ruta = input("Ingresa la ruta del archivo CSV: ").strip()

    if not ruta:
        print("Error: No se proporcionó ninguna ruta de archivo.")
        sys.exit(1)


    try:
        ejecutar_pipeline(ruta)
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
