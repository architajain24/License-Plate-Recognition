import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.title("Car Number Plate Detection")

# Load CSV file containing number plate details
plate_details_df = pd.read_csv("number_plate_details.csv")
plate_details_df.set_index('Number Plate', inplace=True)

harcascade = "model/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

cap.set(3, 640) # width
cap.set(4, 480) # height

min_area = 500
count = 0

while True:
    success, frame = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            plate_img = frame[y: y+h, x:x+w]

            # Extract the number plate text using OCR (you may need to install pytesseract and pytesseract.pytesseract)
            # number_plate_text = pytesseract.image_to_string(plate_img)

            # Match the detected number plate with entries in the CSV file
            number_plate_text = "ABC123"  # Example number plate text for demonstration
            if number_plate_text in plate_details_df.index:
                plate_details = plate_details_df.loc[number_plate_text]
                st.write("Detected Number Plate:", number_plate_text)
                st.write("Owner:", plate_details['Owner'])
                st.write("Model:", plate_details['Model'])
                st.write("Color:", plate_details['Color'])

            st.image(plate_img, caption='Number Plate', use_column_width=True)

    st.image(frame, channels="BGR", caption='Processed Video', use_column_width=True)

    if st.button('Save Plate'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", plate_img)
        st.write("Plate Saved as scaned_img_" + str(count) + ".jpg")
        count += 1

    if st.button('Exit'):
        break

cap.release()
cv2.destroyAllWindows()
