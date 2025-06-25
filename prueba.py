import pandas as pd

try:
    import openpyxl
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "openpyxl"])

ruta_csv = r"C:\Users\erika\proyecto_modelo_predictivo_s4p\salidas\dataset_final.csv"
ruta_xlsx = r"C:\Users\erika\proyecto_modelo_predictivo_s4p\salidas\dataset_final.xlsx"

df = pd.read_csv(ruta_csv)
df.to_excel(ruta_xlsx, index=False)

print("¡Conversión completada exitosamente!")


