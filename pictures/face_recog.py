import os
import cv2
import face_recognition as fg

known_images = os.listdir("known")
unknown_images = os.listdir("unknown")
    
known = ["./known/" + filename for filename in known_images]
names = [name[:-4].title() for name in known_images]
known_face_encodings = []

for face in known:
    img = cv2.imread(face, 1)
    face_encodings = fg.face_encodings(img)[0]
    known_face_encodings.append(face_encodings)

for image in unknown_images:
#for i in range(1):
    img = cv2.imread("./unknown/" + image, 1)
    #img = cv2.imread("./unknown/unknown_4.jpg", 1)
    locations = fg.face_locations(img)
    unknow_face_encoding = fg.face_encodings(img, locations)
    for location, un_face_ed in zip(locations, unknow_face_encoding):
        try:
            top, right, bottom, left = location
            img = cv2.rectangle(img, (left, top),
                                (right, bottom),
                                (255,255,255), 2)
            img = cv2.rectangle(img, (right + 1, bottom),
                                (left - 1, bottom + 20),
                                (255,255,255), -1)
            person_name = "Unknown"
            result = fg.compare_faces(known_face_encodings, un_face_ed, tolerance = 0.53)
            if True in result:
                person_name = names[result.index(True)]

            img = cv2.putText(img, person_name, (left + 5, bottom + 15),
                              cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0,0), 1, cv2.LINE_AA) 

        except Exception as err:
            print(err)

        cv2.imshow(image, img)
        #cv2.imshow("unknown_3", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
