import math
import time
import wave
import cv2
import pyaudio
import pylab
import numpy as np
import asynckivy as ak
import uuid
import firebase_admin
import pyrebase
from firebase_admin import firestore
from joblib import  load
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Rectangle
from kivy.uix.screenmanager import Screen
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivymd.font_definitions import theme_font_styles
from kivymd.uix.button import MDFlatButton, MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField


# Fetch the service account key JSON file contents
cred_object = firebase_admin.credentials.Certificate('covid-app-b87ac-firebase-adminsdk-hb0yl-1254aed690.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred_object, {
    'databaseURL': 'https://covid-app-b87ac.firebaseio.com'
})
db = firestore.client()

pyrebase_config = {
    'apiKey': "",
    'authDomain': "covid-app-b87ac.firebaseapp.com",
    'databaseURL': "https://covid-app-b87ac-default-rtdb.firebaseio.com",
    'projectId': "covid-app-b87ac",
    'storageBucket': "covid-app-b87ac.appspot.com",
    'messagingSenderId': "",
    'appId': ""
}

_pyrebase = pyrebase.initialize_app(pyrebase_config)
database = _pyrebase.storage()
Window.size = (390, 600)
catboost_meta_model = load('catboost_meta_model.joblib')
covid_spectrograms_xgboost_model = load('covid_spectrograms_xgboost_model.joblib')
covid_spectrograms_catboost_model_1 = load('covid_spectrograms_catboost_model_1.joblib')
covid_spectrograms_catboost_model_2 = load('covid_spectrograms_catboost_model_2.joblib')
covid_spectrograms_catboost_model_3 = load('covid_spectrograms_catboost_model_3.joblib')
covid_spectrograms_logistic_regression_model = load('covid_spectrograms_logistic_regression_model.joblib')


class Frame(Widget):
    def __init__(self, a):
        super(Frame, self).__init__()
        with self.canvas:
            Color(a[0], a[1], a[2], a[3])
            window_size = Window.size
            size = Window.width
            self.rect = Rectangle(pos=self.pos, size=(10000, Window.height / 6))


class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.screen = Screen()
        self.frame = Frame([0, 0, 0, .04])
        self.proof = str()
        self.status = str()
        self.text_field = MDTextField(
            # h_align='center',
            pos=(0, 0),
            size_hint=(.61, .045),
            halign='center',
            hint_text="Enter Location to medical result",
            mode='rectangle',
            pos_hint={'center_x': 0.5, "center_y": 0.17},
            helper_text="path to medical result stored on local file system",
            helper_text_mode="on_focus",
        )
        self.yes_button = MDFlatButton(
            text="YES",
            elevation=0,
            theme_text_color='Custom',
            text_color=(0, 0, 0, .82),
            line_color=(0, 0, 0, .38),
            pos_hint={"center_x": 0.3, "center_y": 0.26},
            font_size=(14),
            size_hint=(.26, .08),
            on_release=self.empty_no
        )
        self.no_button = MDFlatButton(
            text="NO",
            elevation=0,
            theme_text_color='Custom',
            text_color=(0, 0, 0, .82),
            line_color=(0, 0, 0, .38),
            pos_hint={"center_x": 0.7, "center_y": 0.26},
            font_size=(14),
            size_hint=(.26, .08),
            on_release=self.empty_yes
        )
        self.diagnosis = list()
        LabelBase.register(
            name="JetBrainsMono",
            fn_regular="SourceCodePro-Light.ttf")
        theme_font_styles.append('JetBrainsMono')
        self.theme_cls.font_styles["JetBrainsMono"] = [
            "JetBrainsMono",
            36,
            False,
            1.75,
        ]
        LabelBase.register(
            name="JetBrainsMonoRegular",
            fn_regular="SourceCodePro-Regular.ttf")
        theme_font_styles.append('JetBrainsMonoRegular')
        self.theme_cls.font_styles["JetBrainsMonoRegular"] = [
            "JetBrainsMonoRegular",
            36,
            False,
            1.75,
        ]

        self.diagnosis_text = MDLabel(
            text="",
            theme_text_color="Custom",
            text_color=(1, 0, 0, .63),
            halign='center',
            font_style='JetBrainsMonoRegular'
        )
        self.record_button = MDIconButton(
            icon='record.png',
            pos_hint={"center_x": 0.5, "center_y": 0.8},
            user_font_size="164sp",
            # on_release=self.record_wave,
        )
        self.theme_cls.font_styles["JetBrainsMono"] = [
            "JetBrainsMono",
            16,
            False,
            0.75,
        ]
        self.record_text = MDLabel(
            text='Tap to record',
            text_color=(10, 10, 1),
            halign='center',
            pos_hint={"center_y": 0.68},
            font_style='JetBrainsMono',
        )

    # def record_wave(self, arg):
    # chunk = 1024
    # sample_format = pyaudio.paInt16
    # channels = 2
    # fs = 44100
    # RECORD_SECONDS = 3
    # filename = 'sample.wav'
    #
    # self.record_text.text = 'Recorded!'
    # time.sleep(1)
    # p = pyaudio.PyAudio()
    # stream = p.open(format=sample_format, channels=channels,
    #                 rate=fs, input=True, output=True,
    #                 frames_per_buffer=chunk)
    # print('recording...')
    # frames = []
    # for i in range(0, int(fs / chunk * RECORD_SECONDS)):
    #     data = stream.read(chunk)
    #     frames.append(data)
    # stream.stop_stream()
    # stream.close()
    # p.terminate()
    # wf = wave.open(filename, 'wb')
    # wf.setnchannels(channels)
    # wf.setsampwidth(p.get_sample_size(sample_format))
    # wf.setframerate(fs)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    # self.preprocess_and_predict()

    def update_diagnosis(self):
        self.diagnosis_text.text = self.diagnosis[0]
        self.diagnosis_text.text_color = self.diagnosis[1]
        self.frame = Frame(self.diagnosis[2])

    def upload_result(self, arg):
        self.proof = self.text_field.text
        self.upload_images('spectrogram.png', self.proof, self.status)

    def empty_no(self, arg):
        self.no_button.text = ''
        self.yes_button.text = 'YES'
        self.status = 'Correct'

    def empty_yes(self, arg):
        self.yes_button.text = ''
        self.no_button.text = 'NO'
        self.status = 'Incorrect'

    def upload_images(self, cough, proof, status):
        cough_reference = database.child(str(uuid.uuid4())).put(cough)
        print(cough_reference)
        url = database.put(proof)
        proof_reference = database.child(str(uuid.uuid4())).put(proof)
        doc_ref = db.collection(status).add({
            u'cough': cough_reference['name'],
            u'proof': proof_reference['name'],
        })

    async def preprocess_and_predict(self):
        wav = wave.open('sample.wav', 'r')
        frames = wav.readframes(-1)
        sound_info = pylab.fromstring(frames, 'Int16')
        frame_rate = wav.getframerate()
        wav.close()
        pylab.figure(num=None, figsize=(5, 5))
        pylab.subplot(111)
        pylab.specgram(sound_info, Fs=frame_rate)
        pylab.savefig('spectrogram.png')
        spectrogram = np.expand_dims(
            cv2.cvtColor(cv2.resize(cv2.imread('spectrogram.png'), (331, 331)), cv2.COLOR_BGR2GRAY), axis=2)
        spectrogram = np.expand_dims(spectrogram.flatten(), axis=0)
        time_ = time.time()
        meta_model_x = np.hstack((
            covid_spectrograms_catboost_model_1.predict_proba(spectrogram[:, :36520]),
            covid_spectrograms_catboost_model_2.predict_proba(spectrogram[:, 36521:73040]),
            covid_spectrograms_catboost_model_3.predict_proba(spectrogram[:, 73040:]),
            covid_spectrograms_logistic_regression_model.predict_proba(spectrogram),
            covid_spectrograms_xgboost_model.predict_proba(spectrogram),
        ))
        prediction = catboost_meta_model.predict_proba(meta_model_x)[0][0]
        print(catboost_meta_model.predict_proba(meta_model_x))
        # print(time.time() - time_)
        prediction = math.floor(prediction) if prediction < 0.76 else math.ceil(prediction)
        predictions = [['NEGATIVE', (.2, .6, 1, .74), (.2, .6, 1, .15)], ['POSITIVE', (1, 0, 0, .63), (1, .5, .5, .15)]]
        self.diagnosis = predictions[prediction]
        self.update_diagnosis()
        print(prediction)
        self.screen.clear_widgets()
        self.screen.add_widget(self.frame)
        self.screen.add_widget(self.no_button)
        self.screen.add_widget(self.yes_button)
        self.screen.add_widget(self.record_button)
        self.screen.add_widget(self.text_field)
        self.screen.add_widget(self.record_text)
        self.screen.add_widget(self.diagnosis_text)
        self.screen.add_widget(
            MDFlatButton(
                text="SUBMIT", elevation=0, theme_text_color='Custom',
                text_color=(0, 0, 0, .82), line_color=(0, 0, 0, .38),
                pos_hint={'center_x': 0.5, "center_y": 0.07}, font_size=(14),
                size_hint=(.67, .08),
                on_press=self.upload_result
            )
        )
        self.screen.add_widget(
            MDLabel(
                text=f"Were you actually {self.diagnosis[0].lower()}?",
                text_color=(1, 1, 1),
                halign='center',
                pos_hint={"center_y": 0.37},
                font_style='JetBrainsMono',
            )
        )

    def build(self):
        self.screen.add_widget(self.frame)
        self.screen.add_widget(self.record_button)
        self.theme_cls.font_styles["JetBrainsMono"] = [
            "JetBrainsMono",
            16,
            False,
            0.75,
        ]

        self.screen.add_widget(self.record_text)
        return self.screen


main_app = MainApp()


async def diagnose(button, text):
    await ak.event(button, 'on_press')
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100
    RECORD_SECONDS = 3
    filename = 'sample.wav'
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=channels,
                    rate=fs, input=True, output=True,
                    frames_per_buffer=chunk)
    frames = []
    async with ak.fade_transition(text):
        text.text = 'Recording in 3...'
    await ak.sleep(1)
    async with ak.fade_transition(text):
        text.text = '2'
    await ak.sleep(1)
    async with ak.fade_transition(text):
        text.text = '1'
    await ak.sleep(1)
    async with ak.fade_transition(text):
        text.text = 'Recording ...'
    for i in range(0, int(fs / chunk * RECORD_SECONDS)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    text.text = 'Recorded'
    async with ak.fade_transition(text):
        text.text = 'Making Diagnosis'
    await main_app.preprocess_and_predict()
    async with ak.fade_transition(text):
        text.text = 'Done!'
    ak.start(diagnose(main_app.record_button, main_app.record_text))

ak.start(diagnose(main_app.record_button, main_app.record_text))
main_app.run()
