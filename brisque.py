# file:        brisque.py
# author:      Artur Rodrigues Rocha Neto
# email:       artur.rodrigues26@gmail.com
# created:     21/01/2021
# description: Implementação do algoritmo BRISQUE.

import cv2
import numpy as np
from itertools import chain
from scipy.special import gamma

def image_preproc_mscn(img):
    """
    image_preproc_mscn(img)
    Normalização de constraste por diferença de média e variância.
    
        Parâmetros
        ----------
            img: a imagem a ser normalizada.
        Retorno
        -------
            mscn: imagem normalizada.
    """
    
    kernel = (7, 7)
    sval = 1.66666667
    
    blurred = cv2.GaussianBlur(img, kernel, sval, borderType=cv2.BORDER_CONSTANT)
    blurred_sq = blurred * blurred
    
    sigma = cv2.GaussianBlur(img * img, kernel, sval, borderType=cv2.BORDER_CONSTANT)
    sigma = np.sqrt(np.abs(sigma - blurred_sq)) + (1.0 / 255.0)
    
    mscn = (img - blurred) / sigma
    
    return mscn

def gdd_fit(img):
    """
    gdd_fit(img)
    Ajusta as intensidades de uma imagem a uma distribuição gaussiana geral simétrica.
    
    Parâmetros
        ----------
            img: a imagem a ser ajustada.
        Retorno
        -------
            alpha: atributo de forma da distribuição.
            std: variância.
    """
    
    gam = np.arange(0.2, 10.001, 0.001)
    r_gam = (gamma(1.0 / gam) * gamma(3.0 / gam) / (gamma(2.0 / gam)**2))
    
    std_sq = np.mean(img**2)
    std = np.sqrt(std_sq)
    mu = np.mean(np.abs(img))
    rho = std_sq / (mu**2)
    
    diff = np.abs(rho - r_gam)
    index = np.argmin(diff)
    alpha = gam[index]
    
    return alpha, std

def agdd_fit(img):
    """
    agdd_fit(img)
    Ajusta as intensidades de uma imagem a uma distribuição gaussiana geral assimétrica.
    
        Parâmetros
        ----------
            img: a imagem a ser ajustada.
        Retorno
        -------
            upsilon: atributo de forma da distribuição.
            left_std: variância à esquerda.
            right_std: variância à direita.
    """
    
    gam = np.arange(0.2, 10.001, 0.001)
    r_gam = ((gamma(2.0 / gam)) ** 2) / (gamma(1.0 / gam) * gamma(3.0 / gam))
    
    left_std = np.sqrt(np.mean((img[img < 0])**2))
    right_std = np.sqrt(np.mean((img[img > 0])**2))
    gamma_hat = left_std / right_std
    rhat = (np.mean(np.abs(img)))**2 / np.mean(img**2)
    rhat_norm = (rhat * ((gamma_hat**3) + 1) * (gamma_hat + 1)) / (((gamma_hat**2) + 1)**2)
    
    diff = (r_gam - rhat_norm)**2
    index = np.argmin(diff)
    upsilon = gam[index]
    
    return upsilon, left_std, right_std

def brisque_features(img):
    """
    brisque_features(img)
    Calcula os atributos BRISQUE de uma imagem (inspiração: https://github.com/bukalapak/pybrisque)
    
        Parâmetros
        ----------
            img: imagem de onde extrair os atributos.
        Retorno
        -------
            features: atributos baseados em ajuste de gaussianas.
    """
    
    scales = 2
    features = []
    img_src = np.copy(img)
    
    for scale in range(scales):
        mscn = image_preproc_mscn(img_src)
        
        alpha, std = gdd_fit(mscn)
        features.extend([alpha, std**2])
        
        shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]
        for shift in shifts:
            shifted_mscn = np.roll(np.roll(mscn, shift[0], axis=0), shift[1], axis=1)
            pair = np.ravel(mscn, order="F") * np.ravel(shifted_mscn, order="F")
            upsilon, left_var, right_var = agdd_fit(pair)
            
            const = np.sqrt(gamma(1.0 / upsilon)) / np.sqrt(gamma(3.0 / upsilon))
            eta = (right_var - left_var) * (gamma(2.0 / upsilon) / gamma(1.0 / upsilon)) * const
            
            features.extend([upsilon, eta, left_var**2, right_var**2])
        
        img_src = cv2.resize(img_src, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    
    return features

def brisque_rgb_features(img):
    """
    brisque_rgb_features(img)
    Calcula os atributos BRISQUE de cada canal RGB de uma imagem.
    
        Parâmetros
        ----------
            img: imagem RBG completa de onde extrair os atributos.
        Retorno
        -------
            features: concatenação de atributos baseados em ajuste de gaussianas de cada canal da imagem.
    """
    
    features = [brisque_features(img[:,:,i] / 255.0) for i in range(3)]
    features = list(chain.from_iterable(features))
    
    return features

