import simpy
import random
import pandas as pd
import numpy as np

# Lista para almacenar los tiempos de ejecución de cada proceso
tiempos_ejecucion = []

def proceso(env, nombre, cpu, memoria, cantidad_memoria, total_instrucciones, velocidad_cpu):
    global tiempos_ejecucion
    tiempo_inicio = env.now
    
    yield memoria.get(cantidad_memoria)

    while total_instrucciones > 0:
        with cpu.request() as req:
            yield req
            instrucciones_ejecutadas = min(total_instrucciones, velocidad_cpu)
            total_instrucciones -= instrucciones_ejecutadas
            yield env.timeout(1)

            if total_instrucciones > 0 and random.random() < 0.1:
                yield env.timeout(2)

    tiempo_final = env.now
    tiempos_ejecucion.append(tiempo_final - tiempo_inicio)
    
    yield memoria.put(cantidad_memoria)

def configuracion_simulacion(env, num_procesos, intervalo, capacidad_memoria, velocidad_cpu, num_cpus):
    cpu = simpy.Resource(env, capacity=num_cpus)
    memoria = simpy.Container(env, init=capacidad_memoria, capacity=capacidad_memoria)

    for i in range(num_procesos):
        tiempo_llegada = random.expovariate(1.0 / intervalo)
        env.process(proceso(env, f'Proceso {i}', cpu, memoria, random.randint(1, 10), random.randint(1, 10), velocidad_cpu))
        yield env.timeout(tiempo_llegada)

# Ejecutar simulaciones con diferentes configuraciones
def ejecutar_simulaciones(intervalos, capacidades_memoria, velocidades_cpu, num_cpus_list, num_procesos_list):
    resultados = []
    for intervalo in intervalos:
        for capacidad_memoria in capacidades_memoria:
            for velocidad_cpu in velocidades_cpu:
                for num_cpus in num_cpus_list:
                    for num_procesos in num_procesos_list:
                        env = simpy.Environment()
                        env.process(configuracion_simulacion(env, num_procesos=num_procesos, intervalo=intervalo,
                                                            capacidad_memoria=capacidad_memoria, velocidad_cpu=velocidad_cpu,
                                                            num_cpus=num_cpus))
                        env.run()
                        promedio = np.mean(tiempos_ejecucion) if tiempos_ejecucion else 0
                        desviacion_std = np.std(tiempos_ejecucion) if tiempos_ejecucion else 0
                        resultados.append({'Intervalo': intervalo, 'Capacidad Memoria': capacidad_memoria,
                                        'Velocidad CPU': velocidad_cpu, 'Num CPUs': num_cpus,
                                        'Número de Procesos': num_procesos, 'Promedio Tiempo Ejecución': promedio,
                                        'Desviación Estándar': desviacion_std})
                        tiempos_ejecucion.clear()  # Limpiar para la próxima simulación
    
    # Convertir los resultados en un DataFrame y guardar en un archivo CSV
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("C:\\Users\\nicol\\OneDrive\\Documents\\UVG\\Tercer Semestre\\Algoritmos y Estructura de Datos\\HT5_AED\\promedios_procesos.csv", index=False)
    print("Simulación completada. Los resultados han sido guardados.")

# Parámetros para las simulaciones variadas
intervalos = [10, 5, 1]
capacidades_memoria = [100, 200]
velocidades_cpu = [3, 6]
num_cpus_list = [1, 2]
num_procesos_list = [25, 50, 100, 150, 200]

# Ejecutar las simulaciones variadas
ejecutar_simulaciones(intervalos, capacidades_memoria, velocidades_cpu, num_cpus_list, num_procesos_list)
