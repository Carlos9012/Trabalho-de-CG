#instalando as dependências necessárias: pip install matplotlib numpy
import numpy as np
import utils2 as fgl
import matplotlib.pyplot as plt
from objects import Cano, CanoCurvo, Cilindro, LinhaReta, Paralelepipedo
from utils import  plotar, compor_cena_varios, mostrar_cena_camera_3D, projetar_perspectiva_2d, rasterizar_cena, visualizar_rasterizacao


if __name__ == '__main__':
    cano = Cano(1.0, 5.0, 0.1)
    cano_curvo = CanoCurvo(0.5, 5.0, 0.1, n_segmentos=32)
    cilindro = Cilindro(1.2, 3.0)
    linha4 = LinhaReta()
    paralelepipedo = Paralelepipedo(largura=4, altura=3, espessura=2, n=4, m=3, l=1)

    #Q1
    #plotar(cano, mostrar_malha=True, title="Cano Reto com Malha")
    #plotar(cano_curvo, mostrar_malha=False, cor='salmon', title="Cano Curvo sem Malha")
    #plotar(cilindro, mostrar_malha=True, cor='khaki', title="Cilindro")
    #plotar(linha4, mostrar_malha=False, cor='red',   title="Linha de Tamanho 4")
    #plotar(paralelepipedo, mostrar_malha=True, cor='lightgreen',   title="Paralelepipedo")

    #Q2
    T = 10
    objetos = [
        (cano, T, np.eye(3), np.array([-4, -4, 0]), 'lightblue'),
        (cano_curvo, T, np.eye(3), np.array([4, 4, 0]), 'salmon'),
        (cilindro, T, np.eye(3), np.array([0, 0, 0]), 'khaki'),
        (linha4,   T, np.eye(3), np.array([8,  8, 0]), 'red'),
        (paralelepipedo, T, np.eye(3), np.array([-10,  -10, 0]), 'lightgreen')
    ]
    
    #compor_cena_varios(objetos, mostrar_malha=True, tam_max=10.0,)

    #Q3
    T = 1
    objetos = [
        (cano, T, np.eye(3), np.array([-4, -4, 0]), 'lightblue'),
        (cano_curvo, T, np.eye(3), np.array([4, 4, 0]), 'salmon'),
        (cilindro, T, np.eye(3), np.array([0, 0, 0]), 'khaki'),
        (linha4,   T, np.eye(3), np.array([8,  8, 0]), 'red'),
        (paralelepipedo, T, np.eye(3), np.array([-8,  -8, 0]), 'lightgreen')
    ]
    eye = np.array([0, 0, 10])  # Posição da câmera
    at = np.array([1, 0, 0])       # Ponto de foco
    up = np.array([0, 1, 0])       # Vetor "up"

    #mostrar_cena_camera_3D(objetos, eye, at, up)

    #Q4
    T=1
    objetos = [
        (cano, T, np.eye(3), np.array([-10, -10, 0]), 'lightblue'),
        (cano_curvo, T, np.eye(3), np.array([2, 0, 0]), 'salmon'),
        (cilindro, T, np.eye(3), np.array([0, 0, 0]), 'khaki'),
        (linha4,   T, np.eye(3), np.array([-5,  0, 0]), 'red'),
        (paralelepipedo, T, np.eye(3), np.array([-3,  5, 0]), 'lightgreen')
    ]
     # Projeção em perspectiva dos mesmos objetos
    eye = np.array([5, 10, 10])    # Câmera posicionada a 15 unidades acima da origem
    at = np.array([0, 1, 0])  
    #projetar_perspectiva_2d(objetos, eye, at, fov=60)

    # 5. Função para projetar e rasterizar
    resolucao = (800, 600)
    cena_raster = rasterizar_cena(objetos, eye, at, up, resolucao)
    visualizar_rasterizacao(cena_raster, resolucao)