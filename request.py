import requests
import json
import cv2
import os
from time import sleep


url = "http://34.72.243.5:5000/faster"
headers = {"content-type": "image/jpg"}

vs = cv2.VideoCapture('/Users/kimiak/Desktop/CRCFootage.mp4')

while True:
    count = 0
    ret, image = vs.read()
    cv2.imwrite('/Users/kimiak/Desktop/tigtong.jpg', image)
    image1 = cv2.imread('/Users/kimiak/Desktop/tigtong.jpg')
    _, img_encoded = cv2.imencode(".jpg", image1)
    
    
    response = requests.post(url, data=img_encoded.tostring(), headers=headers)
    predictions = response.json()
    
    for i in range(len(predictions)):
        count += 1
        cv2.rectangle(image1, 
                      predictions[i][0], 
                      predictions[i][1],
                      color=(0, 255, 0), 
                      thickness= 3) 
    print(count)
    os.remove('/Users/kimiak/Desktop/tigtong.jpg')
    if cv2.waitKey(25) & 0xFF == ord('q'):
        vs.release()
        cv2.destroyAllWindows()
        break