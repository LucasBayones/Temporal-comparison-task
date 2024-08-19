#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:29:40 2024

@author: lucasbayones
"""



import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import shutil
from os.path import join as pjoin
from mpl_toolkits.mplot3d import axes3d
import matplotlib
from matplotlib import style
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import math
import pandas as pd
import numba
from numba import jit
from tqdm import tqdm
import time
from sklearn import preprocessing


def importarArchivo(namefile, delimiter, iniciar):
    with open(namefile, 'r') as file:
        Archivo = file.readlines()
    
    datos = []
    for line in Archivo:
        if line.strip():  # Evitar líneas vacías
            tmp = line.strip().split(delimiter)
            datos.append([float(i) for i in tmp[iniciar::]])
    return datos



@jit(nopython=True)
def firingRate(spk2, t_ini=-8, t_fin=10, win=0.2, stp=0.05, **kwargs):
    """
    Esta función calcula la tasa de disparo, toma la variable spk2 
    que son los datos de los tiempos de los potenciales de acción
    convertidos de una lista de str a un array de numpy. Luego esta
    función separa todos los ensayos del archivo y convierte esos datos
    a float. Seguir detallando..
    """
    
    tam1=int(np.floor(((t_fin-t_ini)/stp)-(win/stp)))+1
    sumatoria=np.zeros((1,tam1),dtype=np.float32)
    tiempo = np.zeros(tam1)
    col=0
    t_ini2=t_ini
    while t_ini2+win <= t_fin:
        tiempo[col] = t_ini + col*stp + win # Actualización de tiempo al inicio de la iteración
        sumatoria[0,col] = np.sum((spk2 >= t_ini2) * ((spk2 < (t_ini2 + win)))) #mantenemos el reglon y llenamos por columna
        t_ini2 = t_ini + col*stp + stp
        col += 1
    return sumatoria/win, tiempo #Devuelvo la tasa de disparo, que es la matriz de tasa (el conteo de espigas) dividido por la ventana.





carpeta_busqueda = "/Volumes/TOSHIBA EXT/Neuronas VPC Etiquetadas" #donde vamos a buscar las neuronas

tipo_set = "OCPSLA" #el tipo de set por el cual estan ordenadas las neuronas

directorio_neuronas_etiquetadas = os.listdir(carpeta_busqueda)

while(True):
    
    print("Menú de Alineación: \n")
    print("1: Inicio del primer estímulo")
    print("2: Fin del primer estímulo")
    print("3: Inicio del segundo estímulo")
    print("4: Fin del segundo estímulo")
    
    tiempo_deseado = int(input("Presiona la tecla de selección "))
    
    if tiempo_deseado in range(1,5):
        break
    else:
        print("Selección incorrecta")
    
tasas_neuronas = []    
clases_neuronas = []

for carpeta_i in directorio_neuronas_etiquetadas:
    
    if tipo_set in carpeta_i:
        
        print(carpeta_i)
        
        set_a_utilizar = pjoin(carpeta_busqueda,carpeta_i)
        neuronas = os.listdir(set_a_utilizar)
        
        indice_neurona=-1
        
        for neurona_i in tqdm(neuronas):
            if neurona_i.find("neu")!= -1 and neurona_i.find(".csv") and neurona_i.find("._")==-1:
                print(neurona_i)
                
                psico_neurona = np.loadtxt(pjoin(carpeta_busqueda,carpeta_i,neurona_i),delimiter=",",usecols=range(31)) #hasta la columna 30 es psicometria
                
                
                aciertos = np.where(psico_neurona[:,4]==1)[0]
               
                
                psico_neurona_aciertos = psico_neurona[aciertos,:]
                
                # if len(psico_neurona_aciertos)<70:
                #     continue
                
                clases, ensayos_validos = np.unique(psico_neurona_aciertos[:,1], return_counts=True)
                clases = np.int32(clases)
             
                             
                if len(clases)==0:
                    continue
                
                if len(ensayos_validos)<5:
                    continue
                
                if len(clases)!=14:
                    continue
                
                espigas_neurona = importarArchivo(pjoin(carpeta_busqueda,carpeta_i,neurona_i),",",31) #de la 31 en adelante son espigas
                espigas_neurona_aciertos = [espigas_neurona[i] for i in aciertos]
                
                indice_neurona+=1
                
                clases_neuronas.append(clases)
                
                tiempo_espigas_alineadas = []
                tasas_neurona_clase = []
                
                
                # datos_bineados_todas_las_clases = []
                
                for clase_i in clases:
                    
                    posicion_ensayos = np.nonzero(psico_neurona_aciertos[:,1]==clase_i)[0] #regresa solo los indices
                    
                    
                    tasa_temporal = np.empty((len(posicion_ensayos),1181)) #tasa por neurona, tenemos que guardarla afuera
                    
                    tiempo_espigas_alineadas_clase = []
                    tiempo_espigas_alineadas.append(tiempo_espigas_alineadas_clase)
                    
                    for indice, ensayo_i in enumerate(posicion_ensayos):
                        
                                                                        
                        if tiempo_deseado==1:
                                                        
                            alinear_tiempo_deseado = espigas_neurona_aciertos[ensayo_i] - psico_neurona_aciertos[ensayo_i,24] + psico_neurona_aciertos[ensayo_i,25]
                            
                        if tiempo_deseado==2:
                            
                            alinear_tiempo_deseado = espigas_neurona_aciertos[ensayo_i] - psico_neurona_aciertos[ensayo_i,25] + psico_neurona_aciertos[ensayo_i,25]
                            
                        if tiempo_deseado==3:
                            
                            alinear_tiempo_deseado = espigas_neurona_aciertos[ensayo_i] - psico_neurona_aciertos[ensayo_i,26] + psico_neurona_aciertos[ensayo_i,25]
                            
                        if tiempo_deseado==4:
                            
                            alinear_tiempo_deseado = espigas_neurona_aciertos[ensayo_i] - psico_neurona_aciertos[ensayo_i,27] + psico_neurona_aciertos[ensayo_i,25]
                            
                        
                                                                       
                        tiempo_espigas_alineadas_clase.append(alinear_tiempo_deseado)
                        
                        tasa_temporal[indice,:], tiempo = firingRate(alinear_tiempo_deseado,-2,10,0.2,0.01) #corregido, porque antes le pasaba "espigas neurona" directamente.
                        
                                                                
                    tasas_neurona_clase.append(tasa_temporal)
                    
                    
            tasas_neuronas.append(tasas_neurona_clase)                            
                    
            
                    
def promedio_total_cada_neurona_clases (tasas_neuronas):
            
    promedio_totales_neuronas = [] #promedio de tasa de cada neurona a traves de todas las condiciones         
                
    for neurona_i in range(len(tasas_neuronas)):
        cada_neurona_individual = tasas_neuronas[neurona_i]
        cada_neurona_individual_stack = np.vstack(cada_neurona_individual)
        promedio_cada_neurona_individual = np.mean(cada_neurona_individual_stack)
        promedio_totales_neuronas.append(promedio_cada_neurona_individual)
    
    return promedio_totales_neuronas

promedio_totales_neuronas = promedio_total_cada_neurona_clases(tasas_neuronas)


# Para el primer estimulo y la memoria juntamos las clases que tienen mismo duracion de primer estimulo y armamos la matriz de tasas

def juntar_clases(tasas_neuronas):
    
    promedios_tasas = []
    
    # Iterar sobre cada neurona
    for neurona in tasas_neuronas:
        # Crear una lista para guardar los promedios de cada clase
        promedios_neurona = []
        # Iterar sobre cada par de clases
        for clase1, clase2 in zip(neurona[::2], neurona[1::2]):
            clase_combinada = np.vstack((clase1, clase2))
            promedio = np.mean(clase_combinada, axis=0)
            promedios_neurona.append(promedio)
        promedios_tasas.append(promedios_neurona)
    
    return promedios_tasas


promedios_tasas_neuronas_por_clase = juntar_clases(tasas_neuronas)


# Convertir la lista de listas de promedios en un array numpy tridimensional
matriz_tasas = np.array(promedios_tasas_neuronas_por_clase)

# Reordenar los ejes para que las clases sean la última dimensión
matriz_tasas = np.transpose(matriz_tasas, (0, 2, 1))




     
# periodo_de_analisis = [0.4,0.6,0.8,1,1.3,1.6,2] #duraciones de lo intervalos

           
periodo_de_analisis = [2]*7 # duraciones de los intervalos en el delay

vector_binario_analisis = [1,1,1,1,1,1,1,1] # Para las 7 clases juntas (1er estimulo o delay)

 
clases_juntas = [1,2,3,4,5,6,7]
   

# De promedio_totales_neuronas es una lista de promedios para cada neurona entre todas las condiciones
promedio_totales_neuronas_array = np.array(promedio_totales_neuronas)

promedio_totales_neuronas_array = promedio_totales_neuronas_array[:, None]


limite_tiempo = []  # Inicializar la lista para guardar los límites de tiempo
matrices_tasas_temporales = []

for clase in range(len(clases_juntas)):
    inicio_estimulo = np.argmin(np.abs(tiempo))+20 #aca calculo los ejes de proyeccion de -50 ms antes que empiecen los estimulos
    final_estimulo = np.argmin(np.abs(tiempo - periodo_de_analisis[clase]))+20 #Hasta 200 ms luego que terminan los estimulos
    
    limite_tiempo.append(final_estimulo)

    # Extraer la sección de la matriz de tasas para el tiempo específico
    matriz_tasas_temporal = np.copy(matriz_tasas[:, inicio_estimulo:final_estimulo, clase])
    matrices_tasas_temporales.append(matriz_tasas_temporal)  
    
    
# Concatenar las matrices temporales a lo largo del eje del tiempo (eje 1)
matrices_tasas_concatenadas = np.concatenate(matrices_tasas_temporales, axis=1)

matrices_tasas_concatenadas_ajustada = matrices_tasas_concatenadas - promedio_totales_neuronas_array   


# Restar el promedio de cada neurona de cada condición en matriz_tasas para hacer la matriz de tasas de proyeccion
matriz_tasas_proyeccion = matriz_tasas - promedio_totales_neuronas_array[:,:,None]


matriz_covarianza = np.cov(matrices_tasas_concatenadas_ajustada)
   
    
#eigenvalues and eigenvectors
    
eigenvalues, eigevectors = np.linalg.eigh(matriz_covarianza)    
    
indices_de_eigenvalues = np.argsort(eigenvalues)

varianza_explicada = np.flip(eigenvalues[indices_de_eigenvalues])

varianza_explicada_sumada = np.sum(varianza_explicada)

porcentaje_varianza_explicada = (varianza_explicada/varianza_explicada_sumada)*100


#porcentaje de varianza explicada escala lineal
plt.plot(indices_de_eigenvalues+1,porcentaje_varianza_explicada, marker="o") #para graficar el porcentaje de varianza explicada segun el PC
plt.xlim(0,10.5)
plt.xticks(range(11))
plt.ylabel("Expl. Variance (%)")
plt.xlabel("component")
plt.grid(True)
plt.savefig("Porcentaje_Varianza_Explicada_lineal" + ".svg")
plt.show()
plt.close()

#porcentaje de varianza explicada escala log
plt.plot(indices_de_eigenvalues[0:150]+1,porcentaje_varianza_explicada[0:150], marker="o") #para graficar el porcentaje de varianza explicada segun el PC
# plt.xlim(0,10.5)
plt.yscale("log")
# plt.xticks(range(11))
plt.ylabel("log (Expl. Variance)")
plt.xlabel("component")
plt.grid(True)
# plt.savefig("Porcentaje_Varianza_Explicada_log" + ".svg")
plt.show()
# plt.close()



pca = eigevectors[:,np.flip(indices_de_eigenvalues)] #ordeno los eigenvectores de mayor a menor varianza explicada

lista_nombres = ["400","600","800","1000","1300","1600","2000"] #para el primer estimulo clases juntas

lista_colores = plt.colormaps['rainbow'](np.linspace(0,1,8))

estilo_linea = ["-","-","-","-","-","-","-"] # las clases juntas en el primer estimulo y delay

indice_color = [0,1,2,3,4,5,6] #clases juntas en el primer estimulo

# limite_tiempo_proyeccion = [220,240,260,280,310,340,380] # esto es para la parte sensorial, si bien los ejes estan calculados en un rango mas amplio
                                                        # la proyeccion la hago solo con la actividad de lo que duran los estimulos

limite_tiempo_proyeccion = [400,400,400,400,400,400,400] #aqui proyecto con el delay cuando alineo al final de los estimulos

# Este es para graficar los primeros tres PCs (PC1, PC2 y PC3)
fig = plt.figure(constrained_layout=True, figsize=(20, 12))  # Ajusta el tamaño de la figura si es necesario
gs = GridSpec(2, 6, figure=fig)  # Configura GridSpec para dos filas y seis columnas

# Crear subplots individuales para PC1, PC2 y PC3
ax_pc1 = fig.add_subplot(gs[0, 0:2])
ax_pc2 = fig.add_subplot(gs[0, 2:4])
ax_pc3 = fig.add_subplot(gs[0, 4:6])

# Crear subplots para diagramas 2D PC1 vs PC2, PC1 vs PC3 y PC2 vs PC3
ax_pc1_vs_pc2 = fig.add_subplot(gs[1, 0:2])
ax_pc1_vs_pc3 = fig.add_subplot(gs[1, 2:4])
ax_pc2_vs_pc3 = fig.add_subplot(gs[1, 4:6])

# Suponiendo que pca, matriz_tasas_proyeccion, limite_tiempo, vector_binario_analisis, etc. están definidos
for l in range(len(limite_tiempo_proyeccion)):
    if vector_binario_analisis[l] == 1:
        pca1 = pca[:, 0].reshape(1, len(pca)) @ matriz_tasas_proyeccion[:, 200:limite_tiempo_proyeccion[l], l] #proyecto desde -50 ms hasta lo que dura cada estimulo
        pca2 = pca[:, 1].reshape(1, len(pca)) @ matriz_tasas_proyeccion[:, 200:limite_tiempo_proyeccion[l], l]
        pca3 = pca[:, 2].reshape(1, len(pca)) @ matriz_tasas_proyeccion[:, 200:limite_tiempo_proyeccion[l], l]
        pca1 = pca1[0] * -1
        pca2 = pca2[0]
        pca3 = pca3[0]

        ax_pc1.plot(tiempo[200:limite_tiempo_proyeccion[l]], pca1, label=lista_nombres[l], color=lista_colores[indice_color[l]], ls=estilo_linea[l])
        ax_pc2.plot(tiempo[200:limite_tiempo_proyeccion[l]], pca2, label=lista_nombres[l], color=lista_colores[indice_color[l]], ls=estilo_linea[l])
        ax_pc3.plot(tiempo[200:limite_tiempo_proyeccion[l]], pca3, label=lista_nombres[l], color=lista_colores[indice_color[l]], ls=estilo_linea[l])
        
        # Agregar bolitas al final de cada proyección individual
        ax_pc1.scatter(tiempo[limite_tiempo_proyeccion[l] - 1], pca1[-1], s=50, color=lista_colores[indice_color[l]], marker='o')
        ax_pc2.scatter(tiempo[limite_tiempo_proyeccion[l] - 1], pca2[-1], s=50, color=lista_colores[indice_color[l]], marker='o')
        ax_pc3.scatter(tiempo[limite_tiempo_proyeccion[l] - 1], pca3[-1], s=50, color=lista_colores[indice_color[l]], marker='o')
        
        ax_pc1_vs_pc2.plot(pca1, pca2, label=lista_nombres[l], color=lista_colores[indice_color[l]], ls=estilo_linea[l])
        ax_pc1_vs_pc2.scatter(pca1[0], pca2[0], s=50, color=lista_colores[indice_color[l]], marker='*')
        ax_pc1_vs_pc2.scatter(pca1[-1], pca2[-1], s=50, color=lista_colores[indice_color[l]], marker='o')

        ax_pc1_vs_pc3.plot(pca1, pca3, label=lista_nombres[l], color=lista_colores[indice_color[l]], ls=estilo_linea[l])
        ax_pc1_vs_pc3.scatter(pca1[0], pca3[0], s=50, color=lista_colores[indice_color[l]], marker='*')
        ax_pc1_vs_pc3.scatter(pca1[-1], pca3[-1], s=50, color=lista_colores[indice_color[l]], marker='o')

        ax_pc2_vs_pc3.plot(pca2, pca3, label=lista_nombres[l], color=lista_colores[indice_color[l]], ls=estilo_linea[l])
        ax_pc2_vs_pc3.scatter(pca2[0], pca3[0], s=50, color=lista_colores[indice_color[l]], marker='*')
        ax_pc2_vs_pc3.scatter(pca2[-1], pca3[-1], s=50, color=lista_colores[indice_color[l]], marker='o')

# Añadir leyendas y títulos
for ax in [ax_pc1_vs_pc2, ax_pc1_vs_pc3, ax_pc2_vs_pc3]:
    ax.legend()

ax_pc1.set_title('PC1')
ax_pc2.set_title('PC2')
ax_pc3.set_title('PC3')

ax_pc1_vs_pc2.set_xlabel('PC1')
ax_pc1_vs_pc2.set_ylabel('PC2')
ax_pc1_vs_pc2.set_title('PC1 vs PC2')

ax_pc1_vs_pc3.set_xlabel('PC1')
ax_pc1_vs_pc3.set_ylabel('PC3')
ax_pc1_vs_pc3.set_title('PC1 vs PC3')

ax_pc2_vs_pc3.set_xlabel('PC2')
ax_pc2_vs_pc3.set_ylabel('PC3')
ax_pc2_vs_pc3.set_title('PC2 vs PC3')
plt.savefig("TICT Delay"+".svg")
plt.show()
plt.close()



fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")

for l in range(len(limite_tiempo)):
    if vector_binario_analisis[l]==1:
        pca1 = pca[:,0].reshape(1, len(pca))@matriz_tasas_proyeccion[:,200:limite_tiempo_proyeccion[l],l] #Aca limito la clase con el ultimo valor
        pca2 = pca[:,1].reshape(1, len(pca))@matriz_tasas_proyeccion[:,200:limite_tiempo_proyeccion[l],l] #De 0 a 13 porque son 14 clases
        pca3 = pca[:,2].reshape(1, len(pca))@matriz_tasas_proyeccion[:,200:limite_tiempo_proyeccion[l],l]
        pca1 = pca1
        pca2 = pca2 
        pca3 = pca3

        ax1.plot_wireframe(pca1,pca2,pca3,label=lista_nombres[l], color=lista_colores[indice_color[l]],ls=estilo_linea[l])
        ax1.plot(pca1[0][0],pca2[0][0],pca3[0][0],marker = "o", markersize= 8, color = lista_colores[indice_color[l]],ls=estilo_linea[l])
        ax1.plot(pca1[-1][-1],pca2[-1][-1],pca3[-1][-1],marker = "s", markersize= 8, color = lista_colores[indice_color[l]],ls=estilo_linea[l])
        ax1.view_init(elev=50.,azim=-210.)
        
        
        # ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        
        
        
        ax1.set_xlabel("PCA1")
        ax1.set_ylabel("PCA2")
        ax1.set_zlabel("PCA3")
    
ax1.legend(loc=2,prop={"size":6})
plt.savefig("TICT Delay concatenando 3D"+".svg")
plt.show()
plt.close()



# #ESTO ES PARA GRAFICARLO CON LAS LINEAS PUNTEADAS.

# # Suponemos que `pca`, `matriz_tasas_proyeccion`, `limite_tiempo`, `vector_binario_analisis`, etc., están definidos correctamente
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # Encontrar la longitud mínima de trayectoria entre las condiciones activadas
# min_length = min(limite_tiempo[l] - 37 for l in range(len(limite_tiempo)) if vector_binario_analisis[l] == 1)

# for l in range(len(limite_tiempo)):
#     if vector_binario_analisis[l] == 1:
#         start_idx = 37
#         end_idx = limite_tiempo[l]
#         pca1 = -1 * (pca[:,0].reshape(1, len(pca)) @ matriz_tasas_proyeccion[:, start_idx:end_idx, l])
#         pca2 = -1 * (pca[:,1].reshape(1, len(pca)) @ matriz_tasas_proyeccion[:, start_idx:end_idx, l])
#         pca3 = -1 * (pca[:,2].reshape(1, len(pca)) @ matriz_tasas_proyeccion[:, start_idx:end_idx, l])

#         # Graficar la parte continua de la trayectoria
#         ax.plot(pca1[0][:min_length], pca2[0][:min_length], pca3[0][:min_length], label=lista_nombres[l], color=lista_colores[indice_color[l]], linestyle=':')
#         # Graficar la parte punteada de la trayectoria
#         if min_length < end_idx - start_idx:
#             ax.plot(pca1[0][min_length:], pca2[0][min_length:], pca3[0][min_length:], color=lista_colores[indice_color[l]], linestyle='-')

#         # Marcadores para el inicio y fin de la trayectoria
#         ax.scatter(pca1[0][0], pca2[0][0], pca3[0][0], color=lista_colores[indice_color[l]], marker='*', s=50)
#         ax.scatter(pca1[0][-1], pca2[0][-1], pca3[0][-1], color=lista_colores[indice_color[l]], marker='o', s=50)

# ax.view_init(elev=15., azim=90.)

# ax.set_xlabel("PCA1")
# ax.set_ylabel("PCA2")
# ax.set_zlabel("PCA3")
# ax.legend(loc='upper left', prop={'size': 6})
# plt.savefig("Poblacion-1er Estimulo VPC OSCISLA-Computo3D"+".svg")
# plt.show()
# plt.close()





# Esto es KiNeT

tv=tiempo[200:400]


def proyectar_y_expandir_pcs(pca, matriz_tasas_proyeccion, limite_tiempo_proyeccion, vector_binario_analisis, num_pcs, max_len=400):
    # Inicializa una lista para contener todas las proyecciones
    proyecciones_pcs = [[] for _ in range(num_pcs)]
    
    # Proyecta la actividad sobre los primeros num_pcs componentes principales
    for l in range(len(limite_tiempo_proyeccion)):
        if vector_binario_analisis[l] == 1:
            for i in range(num_pcs):
                pca_projection = pca[:, i].reshape(1, len(pca)) @ matriz_tasas_proyeccion[:, 200:limite_tiempo_proyeccion[l], l]
                proyecciones_pcs[i].append(pca_projection)
    
    # Función para expandir las proyecciones
    def expandir_proyecciones(proyecciones):
        proyecciones_expandidas = []
        for array in proyecciones:
            current_length = array.shape[1]
            fill_length = max_len - current_length
            if fill_length > 0:
                fill_array = np.full((array.shape[0], fill_length), np.nan)
                expanded_array = np.concatenate((array, fill_array), axis=1)
            else:
                expanded_array = array
            proyecciones_expandidas.append(expanded_array)
        return proyecciones_expandidas
    
    # Expande cada lista de proyecciones a la longitud máxima
    proyecciones_expandidas = [expandir_proyecciones(proyecciones) for proyecciones in proyecciones_pcs]
    
    # Convertir listas de listas de proyecciones a arrays y apilar para formar la matriz final
    proyecciones_arrays = [np.squeeze(np.array(proy), axis=1) for proy in proyecciones_expandidas]
    data = np.stack(proyecciones_arrays)
    
    return data, proyecciones_pcs

# Uso de la función con un número deseado de componentes principales, por ejemplo, 3
num_pcs = 4
data, proyecciones_pcs = proyectar_y_expandir_pcs(pca, matriz_tasas_proyeccion, limite_tiempo_proyeccion, vector_binario_analisis, num_pcs)





class KiNet():
  '''
  KiNet analysis.

  trajectories: PxNxT matrix, where P is the significant PCs of the data,
  N is the number of trajectories/conditions,and T is number of time windows.
  In case of distinct lenghts, T corresponds to the longest trajectory.

  tv: time vector
  n_ref: index of the reference trajectory
  len_ref: lenght of the reference trajectory (same as T when all trajectories
  have the same lenght)
  '''
  def __init__(self, trajectories, tv, n_ref, len_ref):
    self.tra = trajectories
    self.n_tra = self.tra.shape[1]
    self.tv = tv
    self.n_ref = n_ref
    self.len_ref = len_ref

  def calculate_t_i(self):
    self.t_i = np.zeros((self.n_tra, self.len_ref)) # NxT_ref
    for ti in range(self.len_ref):
      for n in range(self.n_tra):
        dif = self.tra[:,n,:] - self.tra[:,self.n_ref,ti].reshape(-1,1)
        self.t_i[n,ti] = np.nanargmin(np.linalg.norm(dif, axis=0))
    self.t_i = self.t_i.astype(np.int64)


  def distances(self):
    self.s_i = np.zeros((self.tra.shape[0], self.n_tra, self.len_ref)) # PxNxT_ref
    for n in range(self.n_tra):
      self.s_i[:,n,:] = self.tra[:,n,self.t_i[n,:]]

    self.D = np.zeros(self.t_i.shape) # NxT_ref
    for ti in range(self.len_ref):
      first_vec = self.s_i[:,self.n_ref,ti] - self.s_i[:,0,ti]

      for n in range(self.n_tra):
        dif = self.s_i[:,self.n_ref,ti] - self.s_i[:,n,ti]
        self.D[n,ti] = np.linalg.norm(dif) * np.sign(dif @ first_vec)

    return self.D

  def distance_diff_times(self, time_indexes, n_com):
    """
      Calculates the distance between trajectories for different times

      time_indexes: array containing the time index of each trajectory to compare.
      n_com: index of the trajectory to which we want to compare.
    """
    self.D_diff_t = np.zeros(self.n_tra)

    for n in range(self.n_tra):
      dif = self.tra[:, n_com, time_indexes[n_com]] - self.tra[:, n, time_indexes[n]]
      self.D_diff_t[n] = np.linalg.norm(dif)
    return self.D_diff_t

  def absolute_speed(self,dt):
    """
    Calculates the absolute speed (distance over time)

    dt: delta t [seconds]
    """
    self.abs_speed = np.zeros((self.n_tra, len(self.tv)-1)) # Nx(T-1)
    for n in range(self.n_tra):
      self.abs_speed[n,:] = [np.sqrt(np.sum((self.tra[:,n,t+1]-self.tra[:,n,t])**2))/dt
                              for t in range(len(self.tv) - 1)]

    return self.abs_speed

  def distance_and_projection(self, time_indexes, n_com = [3,4]):
    """
      Calculates the distance between trajectories for different times and
      projects them into the axis created by the difference of the reference
      trajectories.

      time_indexes: array containing the time index of each trajectory to compare.
      n_com: list with the 2 indices of the trajectories that create the projection vector.
    """
    self.D_diff_p = np.zeros(self.n_tra)
    proyection_v = self.tra[:, n_com[0], time_indexes[n_com[0]]] - \
            self.tra[:, n_com[1], time_indexes[n_com[1]]]
    proyection_v /= np.linalg.norm(proyection_v)
    for n in range(self.n_tra):
      dif = self.tra[:, n_com[0], time_indexes[n_com[0]]] - \
            self.tra[:, n, time_indexes[n]]
      self.D_diff_p[n] = np.dot(dif, proyection_v)

    return self.D_diff_p



n_ref = 3
for ti in range(len(tv)):
  if np.isnan(data[0,n_ref,ti]):
    len_ref = ti
    break

len_ref = 200

KiNet_data = KiNet(data,tv,n_ref,len_ref)



KiNet_data.calculate_t_i()

n_tra = 7

for n in range(n_tra):
    plt.plot(tv[KiNet_data.t_i[n_ref,:]], tv[KiNet_data.t_i[n,:]], label=str(n+1))
plt.legend()
plt.ylabel(r"$t_{i}$")
plt.xlim([0.25, 2.25])
plt.ylim([0.25, 2.25])
plt.xlabel(r"$t_{ref}$")
plt.savefig("TICT_ti_vs_tref.svg")  # Guardar antes de mostrar
plt.show()


D = KiNet_data.distances()
for n in range(n_tra):
    plt.plot(tv[KiNet_data.t_i[n_ref,:]], D[n,:], label=str(n+1))
plt.xlim([0.25, 2.25])
plt.ylim(-130, 130)
plt.legend()
plt.ylabel(r"$D_{i}$")
plt.xlabel(r"$t_{ref}$")
plt.savefig("TICT_distances_vs_tref.svg")  # Guardar antes de mostrar
plt.show()


end_of_stim_ind = [44,64,84,104,134,164,204] #sensorial
Distances = KiNet_data.distance_and_projection(end_of_stim_ind)
plt.scatter(tv[end_of_stim_ind], Distances)
plt.xlabel("lenght of the Stimulus")
plt.ylabel("Distance projected")
# plt.savefig("distances_vs_tref_1000.svg")  # Guardar antes de mostrar
plt.show()


# end_of_stim_ind = [44,64,84,104,134,164,204] #sensorial
end_of_stim_ind = [199,199,199,199,199,199,199] #delay

# Calcula las distancias
Distances = KiNet_data.distance_diff_times(end_of_stim_ind, 3)

# Modifica las distancias según la descripción
for i in range(len(Distances)):
    if i < 3:
        Distances[i] = -abs(Distances[i])
    elif i == 3:
        Distances[i] = 0
    else:
        Distances[i] = abs(Distances[i])

# Grafica los resultados
plt.scatter(tv[end_of_stim_ind], Distances)
plt.ylabel(r"$D_{1000}$")
plt.xlabel(r"Interval (s)")
# plt.savefig("distances_vs_min_interval TICT 1er Interval.svg")  # Guardar antes de mostrar
plt.show()



def gaussian_kernel(length, sigma):
    if length % 2 == 0:
        length += 1  # Asegurarse de que la longitud es impar
    t = np.linspace(-length // 2, length // 2, length)
    kernel = np.exp(-t**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normaliza el kernel
    return kernel

# Parámetros del kernel
sigma = 1 # El sigma controla el ancho del kernel gaussiano
length = 15  # La longitud del kernel, escoge un número impar

# Genera el kernel
kernel = gaussian_kernel(length, sigma)


abs_speeds = KiNet_data.absolute_speed(0.01)


abs_speeds_smoothed = [np.convolve(i, kernel, mode='same') for i in abs_speeds]

abs_speeds_smoothed_stacked = np.vstack(abs_speeds_smoothed)

for n in range(n_tra):
  plt.plot(tv[:-1], abs_speeds_smoothed_stacked[n,:], label=str(n+1))
plt.legend()
plt.ylabel(r"$ Absolute Speed$")
plt.xlabel(r"$t$")
# plt.savefig("absolute_speed TICT 1er Interval.svg")  # Guardar antes de mostrar
plt.show()
















