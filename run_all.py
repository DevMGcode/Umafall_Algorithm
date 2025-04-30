#!/usr/bin/env python3
# run_all.py - Script para ejecutar todos los scripts del detector de caídas en orden

import os
import subprocess
import time
import sys

def print_header(message):
    """Imprime un encabezado formateado para cada paso"""
    print("\n" + "="*80)
    print(f"  {message}")
    print("="*80 + "\n")

def run_command(command, description, exit_on_error=True):
    """Ejecuta un comando y muestra su salida"""
    print_header(description)

    # Ejecutar el comando y mostrar su salida en tiempo real
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Mostrar la salida en tiempo real
    for line in process.stdout:
        print(line, end='')

    # Esperar a que termine el proceso
    process.wait()

    # Verificar si el comando se ejecutó correctamente
    if process.returncode != 0:
        print(f"\n❌ Error al ejecutar: {command}")
        if exit_on_error:
            sys.exit(1)
        else:
            print("Continuando a pesar del error...")
    else:
        print(f"\n✅ Completado: {command}")

def main():
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("UMAFall_Dataset.zip"):
        print("❌ Error: No se encontró el archivo UMAFall_Dataset.zip en el directorio actual.")
        print("Por favor, ejecuta este script desde el directorio raíz del proyecto.")
        sys.exit(1)

    # Verificar que existe la carpeta scripts
    if not os.path.exists("scripts"):
        print("❌ Error: No se encontró la carpeta 'scripts' en el directorio actual.")
        sys.exit(1)

    # Crear directorios necesarios si no existen
    for directory in ["data", "models", "results"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Creado directorio: {directory}")

    # Instalar dependencias (no salir si falla)
    run_command("pip install -r requirements.txt", "Instalando dependencias", exit_on_error=False)

    # Ejecutar scripts en orden
    scripts = [
        ("scripts/train_model.py", "Entrenando modelo con datos del sensor de muñeca"),
        ("scripts/main.py", "Ejecutando detector en una muestra"),
        ("scripts/test_detector.py", "Evaluando rendimiento del detector"),
        ("scripts/visualize_data.py", "Visualizando datos y características")
    ]

    for script, description in scripts:
        if not os.path.exists(script):
            print(f"❌ Error: No se encontró el script {script}")
            sys.exit(1)

        run_command(f"python {script}", description)
        time.sleep(1)  # Pequeña pausa entre scripts

    print_header("¡Todos los scripts se ejecutaron correctamente!")
    print("Resultados disponibles en el directorio 'results/'")
    print("Modelo entrenado disponible en 'models/random_forest_model.pkl'")
    print("Características extraídas disponibles en 'data/'")

if __name__ == "__main__":
    main()