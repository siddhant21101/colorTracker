import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class KivyCamera(Image):
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        success, frame = self.capture.read()
        if success:
            # Converting BGR image to HSV format
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Specifying upper and lower ranges of color to detect in HSV format
            lower = np.array([15, 150, 20])
            upper = np.array([35, 255, 255])
            
            # Masking the image to find our color
            mask = cv2.inRange(img, lower, upper)
            
            # Finding contours in mask image
            mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Finding position of all contours
            if len(mask_contours) != 0:
                for mask_contour in mask_contours:
                    if cv2.contourArea(mask_contour) > 500:
                        x, y, w, h = cv2.boundingRect(mask_contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Drawing rectangle
            
            # Convert the image to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

class ColorTrackingApp(App):
    def build(self):
        return KivyCamera()

if __name__ == '__main__':
    ColorTrackingApp().run()
