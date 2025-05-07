import subprocess

def run_script(script_name):
    print(f"\n⏳ Ejecutando: {script_name}")
    result = subprocess.run(["python", f"scripts/{script_name}"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {script_name} completado exitosamente.\n")
    else:
        print(f"❌ Error al ejecutar {script_name}:\n{result.stderr}")
        exit(1)

if __name__ == "__main__":
    print("🚀 Iniciando pipeline AlgoritmoUmaFall...\n")
    
    run_script("one_extract_data_wrist_filter.py")
    run_script("two_model_creation.py")
    run_script("three_fall_detection.py")
    
    print("\n🎉 Pipeline completo.")
