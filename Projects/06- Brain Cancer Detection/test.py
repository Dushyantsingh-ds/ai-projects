from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

print("Loaded model from disk")
label = ["Benign", "Malign", "Normal"]
path2 = "TestDataset/N_1.jpg"
test_image = image.load_img(path2, target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = loaded_model.predict(test_image)
label2 = label[result.argmax()]
print(label2)