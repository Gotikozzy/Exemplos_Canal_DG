import numpy as np
import cv2

"""
    Este módulo tem o codigo de rastreio de movimento entre duas imagens  
    usa o algoritmo de Optical Flow Lucas Kanade, o algortimo foi recriado 
    usando apenas a biblioteca OpenCV para abstraçoes de complexidade de conversões
    de escala de cinza, este código é apenas para fins didáticos
    não leva em conta, falhas, performance e otmização para cenários reais
    
    Autor: Gotikozzy
    Canal DG: https://youtube.com@detonandogueek
    
    Este código é livre para o uso 
    """
def calcular_fluxo_optico_lucas_kanade(img1, img2, tamanho_janela):
    # Convertendo as imagens para escala de cinza
    img1_cinza = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_cinza = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Gradientes das imagens
    gradiente_x = cv2.Sobel(img1_cinza, cv2.CV_64F, 1, 0, ksize=5)
    gradiente_y = cv2.Sobel(img1_cinza, cv2.CV_64F, 0, 1, ksize=5)
    gradiente_t = img2_cinza - img1_cinza

    # Inicializando os arrays de fluxo
    fluxo_u = np.zeros_like(img1_cinza)
    fluxo_v = np.zeros_like(img1_cinza)

    meio_janela = tamanho_janela // 2
    
    # Iterando sobre a imagem
    for y in range(meio_janela, img1_cinza.shape[0] - meio_janela):
        for x in range(meio_janela, img1_cinza.shape[1] - meio_janela):
            janela_gradiente_x = gradiente_x[y - meio_janela:y + meio_janela + 1, x - meio_janela:x + meio_janela + 1].flatten()
            janela_gradiente_y = gradiente_y[y - meio_janela:y + meio_janela + 1, x - meio_janela:x + meio_janela + 1].flatten()
            janela_gradiente_t = gradiente_t[y - meio_janela:y + meio_janela + 1, x - meio_janela:x + meio_janela + 1].flatten()

            A = np.vstack((janela_gradiente_x, janela_gradiente_y)).T
            b = -janela_gradiente_t

            # Resolvendo A.T * A * v = A.T * b
            nu = np.linalg.pinv(A.T @ A) @ A.T @ b

            fluxo_u[y, x] = nu[0]
            fluxo_v[y, x] = nu[1]
    
    return fluxo_u, fluxo_v

def visualizar_fluxo_optico(img, fluxo_u, fluxo_v, passo=16, limiar=1.0):
    """
    Visualiza o Fluxo Óptico.
    """
    contador_setas = 0
    for y in range(0, img.shape[0], passo):
        for x in range(0, img.shape[1], passo):
            if np.sqrt(fluxo_u[y, x]**2 + fluxo_v[y, x]**2) > limiar:  # Verifica se o movimento é significativo
                ponto_final = (int(x + fluxo_u[y, x]), int(y + fluxo_v[y, x]))
                cv2.arrowedLine(img, (x, y), ponto_final, (0, 255, 0), 1, tipLength=0.3)
                contador_setas += 1
    
    # Adiciona o contador de setas na imagem
    cv2.putText(img, f'Setas desenhadas: {contador_setas}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img

# Exemplo de uso
img1 = cv2.imread(r'C:\Users\gotik\Videos\ms\equilibrio\01.png')
img2 = cv2.imread(r'C:\Users\gotik\Videos\ms\equilibrio\02.png')

fluxo_u, fluxo_v = calcular_fluxo_optico_lucas_kanade(img1, img2, tamanho_janela=5)

img_com_fluxo = visualizar_fluxo_optico(img1.copy(), fluxo_u, fluxo_v)

cv2.imshow('Fluxo Óptico', img_com_fluxo)
cv2.imwrite(r'C:\Users\gotik\Videos\ms\equilibrio\fluxo_optico.jpg', img_com_fluxo)
cv2.waitKey(0)
cv2.destroyAllWindows()
