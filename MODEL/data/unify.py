import json
import os

# Ruta raíz donde están todas las carpetas con datasets
carpeta_raiz = "./raw"

# Lista para guardar todos los datos combinados
datos_combinados = []

# Recorremos recursivamente todas las subcarpetas y archivos
for carpeta_actual, subcarpetas, archivos in os.walk(carpeta_raiz):
    for archivo in archivos:
        if archivo.endswith(".json"):
            ruta_completa = os.path.join(carpeta_actual, archivo)
            with open(ruta_completa, "r", encoding="utf-8") as f:
                try:
                    contenido = json.load(f)
                    if isinstance(contenido, list):
                        datos_combinados.extend(contenido)
                    elif isinstance(contenido, dict):
                        datos_combinados.append(contenido)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error en {ruta_completa}: {e}")

# Guardamos los datos combinados en un solo archivo
with open("unificado.json", "w", encoding="utf-8") as f_out:
    json.dump(datos_combinados, f_out, indent=2, ensure_ascii=False)

print("✅ Archivos combinados exitosamente en 'unificado.json'")
