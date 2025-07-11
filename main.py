#instalando as dependências necessárias: pip install matplotlib numpy
import numpy as np
from objects import Cano, CanoCurvo, Cilindro, LinhaReta4
from utils import  plotar, compor_cena_varios, mostrar_cena_camera_




if __name__ == '__main__':
    cano = Cano(1.0, 5.0, 0.1)
    cano_curvo = CanoCurvo(0.5, 5.0, 0.1, 24,[
            [0, 0, 0],
            [1, 2, 1],
            [3, -1, 2],
            [5, 0, 3]
        ]
    )
    cilindro = Cilindro(1.2, 3.0)
    linha4   = LinhaReta4()

    #plotar(cano, mostrar_malha=True, title="Cano Reto com Malha")

    #plotar(cano_curvo, mostrar_malha=False, cor='salmon', title="Cano Curvo sem Malha")
    
    #plotar(cilindro, mostrar_malha=True, cor='khaki', title="Cilindro")
    #plotar(linha4,   mostrar_malha=False, cor='red',   title="Linha de Tamanho 4")

    objetos = [
        (cano, 1.0, np.eye(3), np.array([-5, 0, 0]), 'lightblue'),
        (cano_curvo, 1.0, np.eye(3), np.array([5, 0, 0]), 'salmon'),
        (cilindro, 1.0, np.eye(3), np.array([0, -4, 0]), 'khaki'),
        (linha4,   1.0, np.eye(3), np.array([0,  4, 0]), 'red')
    ]
    #compor_cena_varios(objetos, mostrar_malha=True)
    camera_pos = np.array([10, 10, 10])     # Origem da câmera
    look_at = np.array([1, 0, 0])           # Para onde a câmera está olhando

    mostrar_cena_camera_(objetos, camera_pos, look_at, up=np.array([0, 1, 0]))