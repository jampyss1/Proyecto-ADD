import sys
import os
import io

import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carga_datos import CargadorDatos, SimuladorNoSQL, WebScraper
from analisis_eda import AnalizadorExploratorio
from preprocesamiento import Preprocesador


class AplicacionBeta:

    def __init__(self):
        self._cargador = CargadorDatos()
        self._fuentes = {}
        self._simulador_nosql = None
        self._scraper = WebScraper()
        self._config_pg = None

    def _mostrar_menu(self):
        print("  Fuentes de datos:")
        print("    1. Cargar archivo CSV")
        print("    2. Cargar archivo TSV")
        print("    3. Cargar archivo JSON")
        print("    4. Conectar a base de datos SQLite")
        print("    5. Visualizar esquema de tablas SQL")
        print()
        print("  Datos avanzados:")
        print("    6. Generar datos NoSQL simulados")
        print("    7. Web Scraping (datos no estructurados)")
        print()
        print("  Análisis:")
        print("    8. Ejecutar EDA sobre fuentes cargadas")
        print()
        print("    0. Salir")
        print()

    def _mostrar_fuentes_cargadas(self):
        if not self._fuentes:
            return
        print(f"  Fuentes en memoria ({len(self._fuentes)}):")
        for i, (nombre, df) in enumerate(self._fuentes.items(), 1):
            print(f"    {i}. {nombre} ({df.shape[0]}×{df.shape[1]})")
        print()

    def _cargar_csv(self):
        ruta = input("  Ruta del archivo .csv: ").strip()
        if not ruta:
            print("  No se proporcionó ruta.")
            return
        df = self._cargador.cargar_desde_ruta(ruta)
        nombre = os.path.basename(ruta)
        self._fuentes[nombre] = df
        self._imprimir_carga_exitosa(nombre, df)

    def _cargar_tsv(self):
        ruta = input("  Ruta del archivo .tsv: ").strip()
        if not ruta:
            print("  No se proporcionó ruta.")
            return
        df = self._cargador.cargar_desde_ruta(ruta)
        nombre = os.path.basename(ruta)
        self._fuentes[nombre] = df
        self._imprimir_carga_exitosa(nombre, df)

    def _cargar_json(self):
        ruta = input("  Ruta del archivo .json: ").strip()
        if not ruta:
            print("  No se proporcionó ruta.")
            return
        df = self._cargador.cargar_desde_ruta(ruta)
        nombre = os.path.basename(ruta)
        self._fuentes[nombre] = df
        self._imprimir_carga_exitosa(nombre, df)

    def _pedir_config_postgres(self):
        if self._config_pg:
            usar = input(f"  Usar conexión anterior ({self._config_pg['user']}@{self._config_pg['host']}:{self._config_pg['port']}/{self._config_pg['database']})? (s/n): ").strip().lower()
            if usar == "s" or usar == "":
                return self._config_pg
        print("\n  Datos de conexión PostgreSQL:")
        host = input("    Host (localhost): ").strip() or "localhost"
        port = input("    Puerto (5432): ").strip() or "5432"
        database = input("    Base de datos: ").strip()
        user = input("    Usuario: ").strip()
        password = input("    Contraseña: ").strip()
        if not database or not user:
            print("  Base de datos y usuario son obligatorios.")
            return None
        config = {
            "host": host,
            "port": int(port),
            "database": database,
            "user": user,
            "password": password
        }
        self._config_pg = config
        return config

    def _conectar_sql(self):
        config = self._pedir_config_postgres()
        if not config:
            return
        tablas = self._cargador.listar_tablas_sql(config)
        if not tablas:
            print("  La base de datos no contiene tablas.")
            return
        print(f"\n  Conexión exitosa a PostgreSQL '{config['database']}'")
        print(f"    Tablas disponibles ({len(tablas)}):")
        for i, tabla in enumerate(tablas, 1):
            print(f"      {i}. {tabla}")
        seleccion = input("\n  Número de la tabla a cargar: ").strip()
        try:
            idx = int(seleccion) - 1
            if idx < 0 or idx >= len(tablas):
                print("  Selección fuera de rango.")
                return
        except ValueError:
            print("  Entrada no válida.")
            return
        tabla = tablas[idx]
        df = self._cargador.cargar_desde_sql(config, tabla)
        nombre = f"PG:{config['database']}.{tabla}"
        self._fuentes[nombre] = df
        self._imprimir_carga_exitosa(nombre, df)

    def _visualizar_esquema_sql(self):
        config = self._pedir_config_postgres()
        if not config:
            return
        tablas = self._cargador.listar_tablas_sql(config)
        if not tablas:
            print("  La base de datos no contiene tablas.")
            return
        print(f"\n  Base de datos PostgreSQL: '{config['database']}'")
        print(f"    Host: {config['host']}:{config['port']}")
        print(f"    Total de tablas: {len(tablas)}")
        pks_cache = {}
        for tabla in tablas:
            pks = self._cargador.obtener_primary_keys(config, tabla)
            pks_cache[tabla] = pks
        for tabla in tablas:
            print(f"\n  ┌─ Tabla: {tabla}")
            columnas_info = self._cargador.obtener_esquema_tabla(config, tabla)
            pks = pks_cache[tabla]
            print(f"  │  Columnas ({len(columnas_info)}):")
            for col_info in columnas_info:
                col_name, data_type, is_nullable, col_default = col_info
                pk = " [PK]" if col_name in pks else ""
                nullable = " (nullable)" if is_nullable == "YES" else ""
                default = f" default={col_default}" if col_default else ""
                print(f"  │    • {col_name} ({data_type}){pk}{nullable}{default}")
            total = self._cargador.contar_registros(config, tabla)
            print(f"  │  Total de registros: {total}")
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=config["host"], port=config["port"],
                    database=config["database"], user=config["user"],
                    password=config.get("password", "")
                )
                df_preview = pd.read_sql_query(f'SELECT * FROM "{tabla}" LIMIT 5', conn)
                conn.close()
                print(f"  │  Primeras 5 filas:")
                for linea in df_preview.to_string(index=False).split("\n"):
                    print(f"  │    {linea}")
            except Exception:
                print(f"  │  (no se pudo obtener preview)")
            print(f"  └{'─' * 50}")

    def _generar_nosql(self):
        self._simulador_nosql = SimuladorNoSQL()
        resumen = self._simulador_nosql.resumen()
        print("\n  Datos NoSQL generados exitosamente.")
        print(f"  Colecciones: {len(resumen)}\n")
        for nombre_col, info in resumen.items():
            print(f"  ┌─ Colección: {nombre_col}")
            print(f"  │  Documentos: {info['total_documentos']}")
            print(f"  │  Campos raíz: {info['campos']}")
            df = self._simulador_nosql.coleccion_a_dataframe(nombre_col)
            print(f"  │  DataFrame aplanado ({df.shape[0]}×{df.shape[1]}):")
            print(f"  │  Columnas: {list(df.columns)}")
            print(f"  │")
            print(f"  │  Primeras 5 filas:")
            for linea in df.head(5).to_string(index=False).split("\n"):
                print(f"  │    {linea}")
            print(f"  └{'─' * 50}")
            clave = f"NoSQL:{nombre_col}"
            self._fuentes[clave] = df
        print(f"\n  Se agregaron {len(resumen)} colecciones a las fuentes cargadas.")

    def _ejecutar_scraping(self):
        url = input("  URL para scraping (Enter para usar Wikipedia SpaceX): ").strip()
        if not url:
            url = "https://en.wikipedia.org/wiki/SpaceX"
        self._scraper.establecer_url(url)
        print(f"\n  Obteniendo datos de: {url}")
        print("  Conectando...")
        extracto = self._scraper.obtener_extracto_html(1200)
        print("\n  ┌─ DATOS NO ESTRUCTURADOS (extracto HTML crudo)")
        print("  │")
        for linea in extracto.split("\n")[:20]:
            print(f"  │  {linea[:100]}")
        print("  │  ...")
        print(f"  └{'─' * 50}")
        tablas = self._scraper.extraer_tablas()
        total = len(tablas)
        print(f"\n  Se encontraron {total} tablas en la página.")
        if total == 0:
            print("  No se encontraron tablas para procesar.")
            return
        tablas_validas = []
        for i, t in enumerate(tablas):
            if len(t) >= 3 and len(t.columns) >= 2:
                tablas_validas.append((i, t))
        if not tablas_validas:
            print("  No se encontraron tablas con datos suficientes.")
            return
        print(f"  Tablas con datos significativos: {len(tablas_validas)}\n")
        for idx_real, (idx_original, tabla) in enumerate(tablas_validas[:5], 1):
            print(f"    {idx_real}. Tabla #{idx_original} ({tabla.shape[0]}×{tabla.shape[1]})")
        seleccion = input("\n  Número de tabla a procesar (Enter = primera): ").strip()
        if not seleccion:
            idx_sel = 0
        else:
            try:
                idx_sel = int(seleccion) - 1
                if idx_sel < 0 or idx_sel >= len(tablas_validas):
                    idx_sel = 0
            except ValueError:
                idx_sel = 0
        idx_original, df_tabla = tablas_validas[idx_sel]
        print(f"\n  ┌─ DATOS ESTRUCTURADOS (tabla procesada)")
        print(f"  │  Origen: {url}")
        print(f"  │  Tabla #{idx_original} — {df_tabla.shape[0]} filas × {df_tabla.shape[1]} columnas")
        print(f"  │")
        print(f"  │  Columnas: {list(df_tabla.columns)}")
        print(f"  │")
        print(f"  │  DataFrame resultante:")
        for linea in df_tabla.head(10).to_string(index=False).split("\n"):
            print(f"  │    {linea}")
        if len(df_tabla) > 10:
            print(f"  │    ... ({len(df_tabla) - 10} filas más)")
        print(f"  └{'─' * 50}")
        nombre_web = f"Web:{os.path.basename(url).split('?')[0]}_tabla{idx_original}"
        self._fuentes[nombre_web] = df_tabla
        print(f"\n  Tabla agregada como '{nombre_web}' a las fuentes cargadas.")

    def _ejecutar_eda(self):
        if not self._fuentes:
            print("\n  No hay fuentes cargadas.")
            return
        print(f"\n  Se ejecutará EDA sobre {len(self._fuentes)} fuente(s):\n")
        for nombre in self._fuentes:
            print(f"    • {nombre}")
        for nombre, df in self._fuentes.items():
            self._ejecutar_eda_individual(df, nombre)
        print(f"\n  EDA completado para todas las fuentes.")

    def _ejecutar_eda_individual(self, df, nombre):
        analizador = AnalizadorExploratorio(df)
        resumen = analizador.resumen_general()
        print(f"\n  {'═' * 50}")
        print(f"  EDA: {nombre}")
        print(f"  {'═' * 50}")
        print(f"    Dimensiones: {resumen['total_registros']} filas × {resumen['total_columnas']} columnas")
        print(f"    Total de valores nulos: {resumen['total_nulos']}")
        print("\n    Tipos de datos por columna:")
        for col, tipo in resumen["tipos_datos"].items():
            print(f"      • {col}: {tipo}")
        print(f"\n    Primeras 5 filas:")
        print(df.head(5).to_string(index=False))
        desc = analizador.estadisticas_descriptivas()
        print(f"\n    Estadísticas descriptivas:")
        print(desc.to_string())
        nulos = analizador.detectar_nulos()
        if nulos:
            print("\n    Columnas con valores nulos (% del total):")
            for col, pct in nulos.items():
                print(f"      {col}: {pct}%")
        else:
            print("\n    Sin valores nulos en el dataset.")
        preprocesador = Preprocesador(df)
        tipos = preprocesador.detectar_tipos_columna()
        print("\n    Clasificación de columnas:")
        for categoria, columnas in tipos.items():
            if columnas:
                print(f"    {categoria.upper()}: {columnas}")
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
        if tipos["categoricas"]:
            print("\n    Distribución de columnas categóricas:")
            for col in tipos["categoricas"]:
                dist = analizador.distribucion_por_categoria(col)
                print(f"       {col}:")
                for cat, freq in list(dist.items())[:10]:
                    print(f"          {cat}: {freq}")
                if len(dist) > 10:
                    print(f"          ... y {len(dist) - 10} categorías más")
        print(f"\n    EDA completado para '{nombre}'.")

    def _imprimir_carga_exitosa(self, nombre, df):
        print(f"\n  Archivo '{nombre}' cargado exitosamente.")
        print(f"    Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        print(f"\n    Primeras 3 filas:")
        print(df.head(3).to_string(index=False))

    def ejecutar(self):
        while True:
            self._mostrar_menu()
            self._mostrar_fuentes_cargadas()
            opcion = input("  Selecciona una opción: ").strip()
            try:
                if opcion == "1":
                    self._cargar_csv()
                elif opcion == "2":
                    self._cargar_tsv()
                elif opcion == "3":
                    self._cargar_json()
                elif opcion == "4":
                    self._conectar_sql()
                elif opcion == "5":
                    self._visualizar_esquema_sql()
                elif opcion == "6":
                    self._generar_nosql()
                elif opcion == "7":
                    self._ejecutar_scraping()
                elif opcion == "8":
                    self._ejecutar_eda()
                elif opcion == "0":
                    break
                else:
                    print("\n  Opción no válida.")
            except FileNotFoundError as e:
                print(f"\n  Error: {e}")
            except ValueError as e:
                print(f"\n  Error: {e}")
            except ConnectionError as e:
                print(f"\n  Error de conexión: {e}")
            except Exception as e:
                print(f"\n  Error inesperado: {e}")
            input("\n  Presiona Enter para continuar...")


if __name__ == "__main__":
    app = AplicacionBeta()
    app.ejecutar()