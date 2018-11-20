# MÓDULOS NORMAIS:
import pandas as pd
import numpy as np
import scipy


# MÓDULOS NECESSÁRIOS PARA A ANÁLISE DE ÁUDIO:
import librosa
print("Termino das importações!")


# VARIÁVEIS GLOBAIS:
sample_rate = 44100
training_path = "C:/Users/Henry Rocha/Documents/GitHub/CDD-Projeto3-Henry/Audios_Train/"
test_path = "C:/Users/Henry Rocha/Documents/GitHub/CDD-Projeto3-Henry/Audios_Test/"


# CLASSES E FUNÇÕES:
class AudioDataExtractor():
    def __init__(self, sample_rate):
        """Declarando o atributo que guarda o Sample Rate usado por todos os áudios análisados:"""
        self.sample_rate = sample_rate

    def set_audio_training_path(self, filepath):
        """Essa função cria o atributo que guarda as direções do diretório em que os áudios de treinamento são mantidos."""
        self.training_path = filepath

    def set_audio_test_path(self, filepath):
        """Essa função cria o atributo que guarda as direções do diretório em que os áudios de treinamento são mantidos."""
        self.test_path = filepath

    def get_waveform(self, audio, target):
        """Essa função utiliza da biblioteca Librosa para retirar do áudio a onde de frequência."""
        if target == "Training":
            path = self.training_path + audio

        elif target == "Test":
            path = self.test_path + audio

        waveform, sampleRate = librosa.load(path, sr=self.sample_rate)
        return waveform

    def doFT(self, waveform):
        """Essa função realiza a Tranformação de Fourier."""
        fourier = scipy.fft(waveform)
        fourier = fourier[:self.sample_rate]
        fourier = np.array(np.abs(fourier))

        return fourier

    def get_fourier_info(self, audio, waveform):
        """Essa função obtem todas as informações possíveis em cima da TF."""
        fourier = self.doFT(waveform)

        data = {}
        data['F_median'] = np.median(fourier)
        data['F_mean'] = np.mean(fourier)
        data['F_1Q'] = np.percentile(fourier, 25)
        data['F_2Q'] = np.percentile(fourier, 75)
        data['F_IQR'] = scipy.stats.iqr(fourier)
        data['F_min'] = np.min(fourier)
        data['F_max'] = np.max(fourier)
        data['F_std'] = np.std(fourier)
        return data

    def get_harmonic(self, audio, waveform):
        """Essa função obtem todas as informações possíveis em cima da harmônica do áudio."""
        print("Começou a analisar o harmônico.")
        harmonic = librosa.effects.harmonic(waveform)
        print("Começou a fazer Fourier com o harmônico.")
        harmonic = self.doFT(harmonic)

        print("Retirando dados do harmônico.")
        data = {}
        data['H_median'] = np.median(harmonic)
        data['H_mean'] = np.mean(harmonic)
        data['H_1Q'] = np.percentile(harmonic, 25)
        data['H_2Q'] = np.percentile(harmonic, 75)
        data['H_IQR'] = scipy.stats.iqr(harmonic)
        data['H_min'] = np.min(harmonic)
        data['H_max'] = np.max(harmonic)
        data['H_std'] = np.std(harmonic)
        return data

    def get_percussive(self, audio, waveform):
        """Essa função obtem todas as informações possíveis em cima da percurssiva do áudio."""
        print("Começou a analisar o percussivo.")
        percussive = librosa.effects.percussive(waveform)
        print("Começou a fazer Fourier com o percussivo.")
        percussive = self.doFT(percussive)

        print("Retirando dados do percussivo.")
        data = {}
        data['P_median'] = np.median(percussive)
        data['P_mean'] = np.mean(percussive)
        data['P_1Q'] = np.percentile(percussive, 25)
        data['P_2Q'] = np.percentile(percussive, 75)
        data['P_IQR'] = scipy.stats.iqr(percussive)
        data['P_min'] = np.min(percussive)
        data['P_max'] = np.max(percussive)
        data['P_std'] = np.std(percussive)
        return data

    def get_spectral_centroid(self, audio, waveform):
        """Essa função obtem todas as informações possíveis em cima do Spectral Centroid."""
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.sample_rate)

        data = {}
        data['SC_median'] = np.median(centroid)
        data['SC_mean'] = np.mean(centroid)
        data['SC_1Q'] = np.percentile(centroid, 25)
        data['SC_2Q'] = np.percentile(centroid, 75)
        data['SC_IQR'] = scipy.stats.iqr(centroid)
        data['SC_min'] = np.min(centroid)
        data['SC_max'] = np.max(centroid)
        data['SC_std'] = np.std(centroid)
        return data

    def get_spectral_flatness(self, audio, waveform):
        """Essa função obtem todas as informações possíveis em cima do Spectral Flatness."""
        flatness = librosa.feature.spectral_flatness(y=waveform)

        data = {}
        data['SF_median'] = np.median(flatness)
        data['SF_mean'] = np.mean(flatness)
        data['SF_1Q'] = np.percentile(flatness, 25)
        data['SF_2Q'] = np.percentile(flatness, 75)
        data['SF_IQR'] = scipy.stats.iqr(flatness)
        data['SF_min'] = np.min(flatness)
        data['SF_max'] = np.max(flatness)
        data['SF_std'] = np.std(flatness)
        return data

    def get_spectral_contrast(self, audio, waveform):
        """Essa função obtem todas as informações possíveis em cima do Spectral Contrast."""
        fourier = np.abs(librosa.stft(waveform))
        contrast = librosa.feature.spectral_contrast(S=fourier, sr=self.sample_rate)

        data = {}
        data['SContr_median'] = np.median(contrast)
        data['SContr_mean'] = np.mean(contrast)
        data['SContr_1Q'] = np.percentile(contrast, 25)
        data['SContr_2Q'] = np.percentile(contrast, 75)
        data['SContr_IQR'] = scipy.stats.iqr(contrast)
        data['SContr_min'] = np.min(contrast)
        data['SContr_max'] = np.max(contrast)
        data['SContr_std'] = np.std(contrast)
        return data

    def extract_features(self, audio, label, target):
        waveform = self.get_waveform(audio, target)

        fourier_info = self.get_fourier_info(audio, waveform)
        centroid_info = self.get_spectral_centroid(audio, waveform)
        flatness_info = self.get_spectral_flatness(audio, waveform)
        contrast_info = self.get_spectral_contrast(audio, waveform)

        # harmonic_info = self.get_harmonic(audio, waveform)
        # percussive_info = self.get_percussive(audio, waveform)

        data = {**fourier_info, **centroid_info, **flatness_info, **contrast_info}

        if target == "Training":
            data["filepath"] = self.training_path + audio

        elif target == "Test":
            data["filepath"] = self.test_path + audio

        data["label"] = label

        return data

    def create_dataset(self, target, starting_index, end_index, filename, full_Loop=False):
        if target == "Training":
            # Carregando o CSV de Treinamento:
            csv = pd.read_csv("train.csv")

        elif target == "Test":
            # Carregando o CSV de Treinamento:
            csv = pd.read_csv("test_sorted.csv")

        # Criando e populando o Dataset:
        dataset = {}

        print("Starting the loop...")
        for i in range(starting_index, end_index):
            audio_name = csv.loc[i, 'fname']
            label = csv.loc[i, 'label']

            dataset[audio_name] = self.extract_features(audio_name, label, target)
            print(audio_name, "nº", i, " --Done!")

            if i % 100 == 0:
                backupFile = pd.DataFrame(dataset)
                backupFile.transpose()
                backupFile.to_csv(path_or_buf=filename + '.csv')

        backupFile = pd.DataFrame(dataset)
        backupFile.transpose()
        backupFile.to_csv(path_or_buf=filename + '.csv')


# PROGRAMA:
# Criando o objeto e configurando o path dos arquivos de treinamento:
DataExtractor = AudioDataExtractor(sample_rate)
DataExtractor.set_audio_training_path(training_path)
DataExtractor.set_audio_test_path(test_path)

# Criando o arquido CSV com as features:
# Treinamento full: 0 a 9473
# Teste full: 0 a 1600
DataExtractor.create_dataset("Test", 0, 1600, "test_features")
