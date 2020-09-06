import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("MNIST_model.h5")

#The program's window
drawing_area = np.ones((400, 400))
#The actual drawing area's color set to 0 (black)
drawing_area[50:350, 50:350] = 0

class Mouse:
	def __init__(self):
		self.start = None
		self.end = None
		self.is_drawing = False

	def draw(self, event, x , y, flags, param):

		if event == cv2.EVENT_LBUTTONDOWN:
			self.is_drawing = True
			self.start = (x,y)

		elif event == cv2.EVENT_MOUSEMOVE:
			if self.is_drawing:
				self.end = (x,y)
				#Draw a line between two points, start and end
				cv2.line(drawing_area, self.start, self.end, 255, 10)
				self.start = self.end
				
		elif event == cv2.EVENT_LBUTTONUP:
			self.is_drawing = False

mouse = Mouse()

cv2.namedWindow("Recognising hand-written digits")
cv2.setMouseCallback("Recognising hand-written digits", mouse.draw)
cv2.putText(drawing_area, "P -predict | C -clear board | Q -quit",
 		   (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), thickness=1)

def get_prediction():
	image = drawing_area[50:350, 50:350]
	#Resize and reshape to (1, 28, 28, 1) for Conv2D on the model
	image = cv2.resize(image, (28,28)).reshape(1, 28, 28, 1)
	image = image/255

	prediction = model.predict(image)
	#Double max() because predict's output is a 2D list
	probability = max(max(prediction))
	probability = "{:.2f}".format(probability*100)+'%'
	#Final prediction of the image
	prediction = np.argmax(prediction)

	return prediction, probability

def main():
	is_using_program = True

	while(is_using_program):
		#Displays 
		cv2.imshow("Recognising hand-written digits", drawing_area)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("p"):
			prediction, probability = get_prediction()
			print("Predicted digit:",prediction,"-->", probability)

		elif key == ord("c"):
			#Set drawing area's color to black 
			drawing_area[50:350, 50:350] = 0

		elif key == ord("q"):
			is_using_program = False
main()

cv2.destroyAllWindows()