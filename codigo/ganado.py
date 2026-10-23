

# %% Uso de Librerias

# Trabajar con rutas relativas en python 
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss




# %% Cargar bases de datos en python usando rutas relativas =========================

# Obtener la ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parents[1]

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "datos"

# Rutas de las bases de datos 
ruta_ganado = DATA_DIR / "serie_hurto_ganado_diaria.csv" # Base de datos de exportaciones




# %% =========================
# PRIMERA SERIE: Hurto de Ganado en Colombia (2019)
# ============================

# Base de datos con la serie importada a python
ganado_base = pd.read_csv(ruta_ganado)

# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print(type(ganado_base))

# Ver primeras y últimas observaciones de la base de datos
print(ganado_base.head()) # Primeras observaciones
print(ganado_base.tail()) # Últimas observaciones 

# Poniendo como índice de la base de datos la columna fecha.
ganado_base.index = pd.to_datetime(ganado_base['Fecha'], format="%Y-%m-%d")
# Eliminando la columna fecha para limpiar la base de datos.
ganado_base = ganado_base.drop(columns=['Fecha'])

# Ver primeras y últimas observaciones de la base de datos, ahora con índice temporal
print(ganado_base.head())
print(ganado_base.tail())




# %% Creación de la serie de tiempo de "ganado"

# La nueva serie de tiempo se va a llamar "ganado_serie" y va a tener valores numéricos
ganado_serie = ganado_base["Cantidad"].copy()
ganado_serie = pd.to_numeric(ganado_serie, errors="coerce")
ganado_serie = ganado_serie.dropna() # Borrar missing values

# Ver el principio y final de la serie de tiempo
print(ganado_serie.head())
print(ganado_serie.tail())

# # El tipo de objeto de la base de datos ahora es un Pandas.Series
print(type(ganado_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print(ganado_serie.describe())


# Rejilla regular cada 14 días desde la primera fecha
rejilla = pd.date_range(start=ganado_serie.index.min(),
                        end=ganado_serie.index.max(),
                        freq="7D")

# Ver el principio y final de la serie de tiempo
print(ganado_serie.head())
print(ganado_serie.tail())

# # El tipo de objeto de la base de datos ahora es un Pandas.Series
print(type(ganado_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print(ganado_serie.describe())




# %% =========================
# Paso 1: Identificación
# ============================

# Gráfica de la serie de tiempo "Hurto de Ganado en Colombia, 2019"
plt.figure(figsize=(10, 5))
plt.plot(ganado_serie, color="#1f4e79", linewidth=1.5)
plt.title("Hurto de Ganado en Colombia, 2019", fontsize=13, fontweight='bold')
plt.xlabel("Fecha")
plt.ylabel("Cabezas Hurtadas")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.7, color='gray')

plt.show()





# %% FAC y FACP de la serie original

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

plot_acf(
    ganado_serie,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0],
    vlines_kwargs={"color": "#1f4e79", "linewidth": 2},
    marker="",
    color = "#1f4e79"
)

axes[0].set_title("FAC hurto de ganado")

plot_pacf(
    ganado_serie,
    lags=24,
    alpha=0.05,
    ax=axes[1],
    vlines_kwargs={"color": "#1f4e79", "linewidth": 2},
    marker="",
    color = "#1f4e79"
)

axes[1].set_title("FACP de la serie original")

plt.tight_layout()
plt.show()








# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(ganado_serie) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(ganado_serie, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie. 



# %% Transformación primeras diferencias de la Serie de Tiempo
ganado_serie_diff = ganado_serie.diff().dropna() # Transformación diferencial

# Gráfica de la serie de tiempo de la "diferencia exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(ganado_serie_diff, color="#c72c11", linewidth=1.5)
plt.title("Serie diferenciada", fontsize=13, fontweight='bold')
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.7, color='gray')
plt.show()


#%%  TEST de raiz unitaria para la serie diferenciada

adf_result = adfuller(ganado_serie_diff) # Test de Dickey Fuller sin intercepto y tendencia determínistica


print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(ganado_serie_diff, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie. 









# %% FAC y FACP de la primera diferencia 

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

plot_acf(
    ganado_serie_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0],
    vlines_kwargs={"color": "#c72c11", "linewidth": 2},
    marker="",
    color = "#c72c11"
)

axes[0].set_title("FAC de la diferencia de la serie")

plot_pacf(
    ganado_serie_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1],
    vlines_kwargs={"color": "#c72c11", "linewidth": 2},
    marker="",
    color = "#c72c11"
)

axes[1].set_title("FACP de la diferencia de la serie")

plt.tight_layout()
plt.show()













# %% =========================
# Paso 2: Estimación
# ============================

# Lo primero es crear un objeto de la clase SARIMAX 
# (Asociado a un MA(1) en éste ejemplo)
# Acá es donde se específica el modelo 
modelo_arima = SARIMAX(
    ganado_serie,
    order=(0, 1, 1),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Objeto tipo "SARIMAX" del paquete statmodels
type(modelo_arima)

#  Acá es donde se realiza la estimación del modelo ARIMA(0,1,1)
estimacion_arima = modelo_arima.fit(disp=False)

# Nota: El método de estimación es máxima verosimilitud 
#       sobre la representación del modelo en un espacio de estados

# Objeto tipo "SARIMAXResultsWrapper"
type(estimacion_arima)

# Se imprimen los resultados principales de la estimación
# E.g. Coeficientes y significancia de los coeficientes
print(estimacion_arima.summary())








# %% ===========================
# Paso 3: Validación de supuestos
# ==============================

# Residuales del modelo
residuales = estimacion_arima.resid.dropna()

# Los residuales son un objeto tipo Pandas.Series
type(residuales)

# Gráfica de los residuales (deberían comportarse como ruido blanco)
plt.figure(figsize=(10, 5))
plt.plot(residuales.iloc[1:-1])
plt.title("Residuales del modelo MA(1)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# Descripción de los residuales 
print(residuales.describe())





# %% FAC y FACP de los residuales

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de los residuales (24 rezagos)")

plot_pacf(
    residuales,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales (24 rezagos)")

plt.tight_layout()
plt.show()






# %% Prueba Ljung-Box

ljung_box = acorr_ljungbox(
    residuales,
    lags=[6, 12, 18, 24],
    return_df=True
)

print("Prueba Ljung-Box")
print(ljung_box)

# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.





# %% FAC y FACP de los residuales al cuadrado

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

plot_acf(
    residuales**2,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0],
    vlines_kwargs={"color": "#ef8d15", "linewidth": 2},
    marker="",
    color = "#ef8d15"
)

axes[0].set_title("FAC de los residuales al cuadrado")

plot_pacf(
    residuales**2,
    lags=24,
    alpha=0.05,
    ax=axes[1],
    vlines_kwargs={"color": "#ef8d15", "linewidth": 2},
    marker="",
    color = "#ef8d15"
)

axes[1].set_title("FACP de los residuales al cuadrado")

plt.tight_layout()
plt.show()








# %% Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuales, nlags=12) 

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

# H0: no hay efectos ARCH.
# Si p-value > 0.05, no se rechaza H0.






# %% Q-Q plot de los residuos

plt.figure(figsize=(6, 6))
probplot(residuales.iloc[1:-1], dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos")
plt.grid(True)
plt.show()






# %% Prueba de normalidad Jarque-Bera

jb_test = jarque_bera(residuales.iloc[1:-1])

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

# H0: los residuos no siguen una distribución normal.
# Si p-value < 0.05, se rechaza H0.






# %% =========================
# PASO 4: Pronóstico
# ============================

# 14 días de pronóstico
pasos = 14
pronostico_obj = estimacion_arima.get_forecast(steps=pasos)

pronostico_puntual = pronostico_obj.predicted_mean
intervalos = pronostico_obj.conf_int(alpha=0.05) # Intervalo de confianza del 95%

# Tabla de pronósticos
tabla_pronostico = pd.DataFrame({
    "Fecha": pronostico_puntual.index,
    "Pronostico_Cabezas": np.round(pronostico_puntual.values, 2),
    "Limite_Inferior_95%": np.round(intervalos.iloc[:, 0].values, 2),
    "Limite_Superior_95%": np.round(intervalos.iloc[:, 1].values, 2)
}).set_index("Fecha")

print("\n" + "="*50)
print(" PRONÓSTICO PARA LOS PRÓXIMOS 14 DÍAS ")
print("="*50)
print(tabla_pronostico)

# --- SOLUCIÓN PARA EL GAP EN LA GRÁFICA ---
ultimo_valor_serie = ganado_serie.tail(1)
ultima_fecha_serie = ultimo_valor_serie.index

# Concatenar el último punto histórico para asegurar la continuidad visual
pronostico_plot = pd.concat([ultimo_valor_serie, pronostico_puntual])
lim_inf_plot = pd.concat([pd.Series(ultimo_valor_serie.values, index=ultima_fecha_serie), intervalos.iloc[:, 0]])
lim_sup_plot = pd.concat([pd.Series(ultimo_valor_serie.values, index=ultima_fecha_serie), intervalos.iloc[:, 1]])

# Gráfico del Pronóstico sin huecos
plt.figure(figsize=(12, 5))
plt.plot(ganado_serie.tail(60), label="Datos Históricos (Últimos 60 días)", color="#1f4e79", linewidth=1.5)
plt.plot(pronostico_plot, label="Pronóstico ARIMA(0,1,1)", color="#e65100", linewidth=2, marker='o', markersize=4)
plt.fill_between(
    pronostico_plot.index,
    lim_inf_plot,
    lim_sup_plot,
    color="#e65100",
    alpha=0.2,
    label="Intervalo de Confianza (95%)"
)
plt.title("Pronóstico del Hurto Diario de Ganado en Colombia - Modelo ARIMA(0,1,1)", fontsize=12, fontweight='bold')
plt.xlabel("Fecha")
plt.ylabel("Cabezas Hurtadas")
plt.legend(loc="upper left")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()