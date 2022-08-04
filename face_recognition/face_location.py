import face_recognition as fr
from PIL import Image, ImageDraw

image = fr.load_image_file('/Users/onyekachukwuokonji/Desktop/face_recognition/elon_musk.jpeg')

locs = fr.face_locations(image, model='cnn')

print(locs)
        
top = locs[0][0] 
right = locs[0][1]
bottom = locs[0][2]
left = locs[0][3]
    
located_face = image[top:bottom, left:right]
    
    
print(located_face.shape)

image_face = Image.fromarray(image)
draw = ImageDraw.Draw(image_face)
draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=5)

image_face.show()

        