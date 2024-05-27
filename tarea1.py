#Importamos las librerias necesarias
import sys
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QComboBox, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QSound
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

#Se crea la clase MainWindow que hereda de QMainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio_data = None
        self.sample_rate = None
        self.processed_audio = None

#Comenzamos a crear la interfaz de usuario
    def initUI(self):
        self.setWindowTitle('Audio Signal Processing')

        #Creamos el layout
        layout = QVBoxLayout()

        #Se crea el boton para cargar el archivo
        self.load_button = QPushButton('Cargar archivo')
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)

        #Se crea el label para seleccionar el filtro
        self.filter_label = QLabel('Seleccionar filtro:')
        layout.addWidget(self.filter_label)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['Pasa bajas', 'Pasa altas', 'Pasa banda'])
        layout.addWidget(self.filter_combo)

        #Se crea el slider para seleccionar la frecuencia de corte
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(100, 5000) #Se establece el rango de la frecuencia de corte
        self.freq_slider.setValue(1000) #Se establece el valor inicial de la frecuencia de corte
        layout.addWidget(self.freq_slider)

        #Se crea el label para mostrar la frecuencia de corte
        self.apply_filter_button = QPushButton('Aplicar Filtro')
        self.apply_filter_button.clicked.connect(self.apply_filter)
        layout.addWidget(self.apply_filter_button)

        #Se crea el boton para aplicar la transformada
        self.apply_transform_button = QPushButton('Aplicar Transformada')
        self.apply_transform_button.clicked.connect(self.apply_transform)
        layout.addWidget(self.apply_transform_button)

        #Se crea el boton para guardar el resultado
        self.save_button = QPushButton('Guardar Resultado')
        self.save_button.clicked.connect(self.save_result)
        layout.addWidget(self.save_button)

        #Se crea el boton para reproducir
        botonreprod = QPushButton('Reproducir Filtrado')
        botonreprod.clicked.connect(self.playFilteredSound)
        layout.addWidget(botonreprod)

        #Se crean los canvas para mostrar las señales
        self.original_signal_canvas = FigureCanvas(plt.figure())
        self.processed_signal_canvas = FigureCanvas(plt.figure())
        layout.addWidget(self.original_signal_canvas)
        layout.addWidget(self.processed_signal_canvas)

        #Se crea el widget central
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

#Se crean los metodos para cargar el archivo
    def load_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Cargar Archivo de Audio", "", "Audio Files (*.wav *.mp3)", options=options)
        if file_name:
            self.audio_data, self.sample_rate = self.load_audio(file_name)
            self.plot_signal(self.audio_data, self.original_signal_canvas)

#Se crea el metodo para aplicar el filtro
    def apply_filter(self):
        if self.audio_data is not None:
            filter_type = self.filter_combo.currentText()
            cutoff_freq = self.freq_slider.value()
            self.processed_audio = self.apply_filter_to_audio(self.audio_data, filter_type, cutoff_freq, self.sample_rate)
            self.plot_signal(self.processed_audio, self.processed_signal_canvas)

#Se crea el metodo para aplicar la transformada
    def apply_transform(self):
        if self.audio_data is not None:
            transform_data = self.apply_transform_to_audio(self.audio_data)
            self.plot_signal(transform_data, self.processed_signal_canvas)

#Se crea el metodo para guardar el resultado
    def save_result(self):
        if self.processed_audio is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Guardar Archivo de Audio", "", "Audio Files (*.wav *.mp3 *.aac)", options=options)
            if file_name:
                self.save_audio(file_name, self.processed_audio, self.sample_rate)

# Función para reproducir el sonido filtrado
    def playFilteredSound(self):
        if self.filtered_signal is not None:
            sf.write(self.temp_filtered_file, self.filtered_signal, self.sr)
            QSound.play(self.temp_filtered_file)


#Se crean los metodos para graficar la señal, cargar el audio, guardar el audio, aplicar el filtro y aplicar la transformada
    def plot_signal(self, data, canvas):
        fig = canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(data)
        canvas.draw()

#Cargar el audio
    def load_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        return y, sr

#Gardar el audio
    def save_audio(self, file_path, y, sr):
        librosa.output.write_wav(file_path, y, sr)

#Aplicar el filtro y aplicar la transformada
    def apply_filter_to_audio(self, y, filter_type, cutoff_freq, sr):
        nyquist = 0.5 * sr
        normal_cutoff = cutoff_freq / nyquist
        if filter_type == 'Pasa bajas':
            b, a = butter(1, normal_cutoff, btype='low', analog=False)
        elif filter_type == 'Pasa altas':
            b, a = butter(1, normal_cutoff, btype='high', analog=False)
        else:  # Pasa banda
            b, a = butter(1, [normal_cutoff, normal_cutoff + 0.1], btype='band', analog=False)
        y_filtered = lfilter(b, a, y)
        return y_filtered

    def apply_transform_to_audio(self, y):
        return np.abs(np.fft.fft(y))

#Se crea la aplicacion
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
