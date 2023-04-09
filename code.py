from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img ,img_to_array

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report,confusion_matrix

my_data_dir = '../input/real-life-industrial-dataset-of-casting-product/casting_data/'

train_path = my_data_dir + 'train/'

test_path = my_data_dir + 'test/'

image_shape = (300,300,1)

batch_size = 32

prediction = model.predict(img.reshape(-1,300,300,1))

if (prediction<0.5):

print("def_front")

cv2.putText(pred_img, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:

print("ok_front")

cv2.putText(pred_img, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
plt.imshow(pred_img,cmap='gray')

plt.axis('off')

plt.show()

prediction = model.predict(img1.reshape(-1,300,300,1))

if (prediction<0.5):

print("def_front")

cv2.putText(pred_img1, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:

print("ok_front")

cv2.putText(pred_img1, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
plt.imshow(pred_img1,cmap='gray')

plt.axis('off')

plt.show()

// go to wiki page for more info
