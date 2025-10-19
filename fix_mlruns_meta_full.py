import os
import yaml

base_dir = "mlruns"
count_fixed = 0
count_skipped = 0
broken_files = []

def fix_meta_file(path):
    global count_fixed, count_skipped
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not data:
            broken_files.append(path)
            return

        # Si no tiene run_uuid pero sí run_id → lo agregamos
        if "run_id" in data and "run_uuid" not in data:
            data["run_uuid"] = data["run_id"]
            with open(path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            print(f"✅ Arreglado: {path}")
            count_fixed += 1
        else:
            count_skipped += 1
    except Exception as e:
        broken_files.append((path, str(e)))


# Recorre todos los meta.yaml dentro de mlruns/
for root, _, files in os.walk(base_dir):
    if "meta.yaml" in files:
        fix_meta_file(os.path.join(root, "meta.yaml"))

print("\n====== RESUMEN ======")
print(f"Archivos corregidos: {count_fixed}")
print(f"Archivos sin cambios: {count_skipped}")
if broken_files:
    print("\n⚠️ Archivos problemáticos:")
    for bf in broken_files:
        print("  -", bf)
else:
    print("✅ Todos los meta.yaml se corrigieron correctamente.")
