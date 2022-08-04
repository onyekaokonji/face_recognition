import pandas as pd
import face_recognition as fr
import cv2
from PIL import Image, ImageDraw,ImageFont

faces_data = pd.read_csv("/Users/onyekachukwuokonji/Desktop/face_recognition/Face_Data .csv")


names = faces_data['Name'].tolist()
occupation = faces_data['Profession'].tolist()
img_file = faces_data['Image File Location'].tolist()

n = len(names)

images = []
face_encodings = []

for i in range(n):
    images.append(fr.load_image_file(img_file[i]))
    face_encodings.append(fr.face_encodings(images[i])[0])


img = cv2.imread("/Users/onyekachukwuokonji/Desktop/face_recognition/elon_musk.jpeg") 
cv2.imwrite('New Image.jpg', img)    

image = fr.load_image_file('New Image.jpg')


def face_detect(img):
    try:
        new_image_encoding = fr.face_encodings(img)[0]
        
    except:
        print('Image Not Found')
        
    found_image = fr.compare_faces(face_encodings, new_image_encoding, tolerance = 0.6)
    
    print(found_image)
    
    index = -1

    for i in range(n):
        if found_image[i]:
            index = i
        
    return(index)

face_detected_index = face_detect(image)
print(face_detected_index)


image_seen = Image.fromarray(image) 
image_draw = ImageDraw.Draw(image_seen)
fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/SignPainter.ttc", size = 100)

if face_detected_index == -1:
    name = 'Face not recognized'
    
else:
    name = names[face_detected_index]
    
x = 30
y = image.shape[0] - 100
image_draw.text((x,y), text=name, font = fnt, fill=(0, 0, 0))
image_seen.show()
    