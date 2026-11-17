import matplotlib
matplotlib.use('Agg')  # Para o GDB não tentar abrir janela
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulando um sinal de EEG
# Taxa de amostragem: 250 Hz (padrão em aparelhos de EEG)
fs = 250  
tempo = np.arange(0, 2, 1/fs)  # 2 segundos de sinal

# Simulando as ondas do cérebro:
# Onda Alpha (10 Hz - relaxamento) + Onda Beta (20 Hz - atenção) + ruído
onda_alpha = 2 * np.sin(2 * np.pi * 10 * tempo)
onda_beta = 0.5 * np.sin(2 * np.pi * 20 * tempo)
ruido = 0.3 * np.random.randn(len(tempo))

sinal_eeg = onda_alpha + onda_beta + ruido

# 2. Limpando o sinal com NumPy (média móvel - filtro simples)
janela = 5
sinal_filtrado = np.convolve(sinal_eeg, np.ones(janela)/janela, mode='same')

# 3. Descobrindo as frequências com FFT
# CORREÇÃO AQUI: deixei o nome igual nas duas linhas -> espectral
espectral = np.fft.rfft(sinal_filtrado)
frequencias = np.fft.rfftfreq(len(sinal_filtrado), 1/fs)
potencia = np.abs(espectral) ** 2  # Antes estava 'espectro', agora 'espectral'

# 4. Calculando a potência nas bandas do EEG
def potencia_banda(f_min, f_max):
    indice = (frequencias >= f_min) & (frequencias <= f_max)
    return np.sum(potencia[indice])

print("=== ANALISE DO SINAL DE EEG ===")
print(f"Potencia Delta (0.5-4 Hz): {potencia_banda(0.5, 4):.2f}")
print(f"Potencia Teta (4-8 Hz): {potencia_banda(4, 8):.2f}")
print(f"Potencia Alpha (8-13 Hz): {potencia_banda(8, 13):.2f}")
print(f"Potencia Beta (13-30 Hz): {potencia_banda(13, 30):.2f}")

# Frequência dominante
freq_dominante = frequencias[np.argmax(potencia)]
print(f"\nFrequencia dominante: {freq_dominante:.1f} Hz -> Onda Alpha (relaxamento)")

# 5. Gráfico para colocar no artigo
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(tempo, sinal_eeg, label='Sinal Original', alpha=0.5)
plt.plot(tempo, sinal_filtrado, label='Sinal Filtrado', color='red')
plt.title('Sinal de EEG - Original vs Filtrado')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude (uV)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(frequencias, potencia)
plt.title('Espectro de Frequencias do EEG')
plt.xlabel('Frequencia (Hz)')
plt.ylabel('Potencia')
plt.xlim(0, 40)

plt.tight_layout()
plt.savefig('eeg.png')  # No GDB usa savefig, não show()
print("Grafico salvo como eeg.png")