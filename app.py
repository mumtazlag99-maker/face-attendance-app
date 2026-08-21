import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, date

DATA_DIR = "data"
ENCODINGS_FILE = os.path.join(DATA_DIR, "encodings.pkl")
ATTENDANCE_DIR = os.path.join(DATA_DIR, "attendance")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

def load_known_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data["names"], data["encodings"]
    return [], []

def save_known_faces(names, encodings):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"names": names, "encodings": encodings}, f)

def get_today_file():
    today_str = date.today().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"{today_str}.csv")

def load_today_attendance():
    path = get_today_file()
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["Name", "Morning", "Afternoon"])

def save_today_attendance(df):
    df.to_csv(get_today_file(), index=False)

def mark_attendance(name):
    df = load_today_attendance()
    now_time = datetime.now().strftime("%H:%M:%S")
    if name in df["Name"].values:
        idx = df.index[df["Name"] == name][0]
        if pd.isna(df.at[idx, "Afternoon"]) or df.at[idx, "Afternoon"] == "":
            df.at[idx, "Afternoon"] = now_time
            save_today_attendance(df)
            return f"Success: {name} ki Afternoon attendance lag gayi ({now_time})"
        else:
            return f"Warning: {name} ki attendance pehle hi lag chuki hai."
    else:
        new_row = {"Name": name, "Morning": now_time, "Afternoon": ""}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_today_attendance(df)
        return f"Success: {name} ki Morning attendance lag gayi ({now_time})"

def get_face_encoding_from_image(image_file):
    image = face_recognition.load_image_file(image_file)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        return None
    return encodings[0]

st.set_page_config(page_title="Face Attendance App", page_icon=":camera:")
st.title("Face Recognition Attendance App")

menu = st.sidebar.radio("Menu", ["Naya Banda Register Karo", "Attendance Lagao", "Attendance Dekho"])
names, encodings = load_known_faces()

if menu == "Naya Banda Register Karo":
    st.header("Naya Banda Register Karo")
    person_name = st.text_input("Naam Likho")
    photo = st.camera_input("Face ki photo lo")
    if st.button("Register Karo"):
        if not person_name:
            st.error("Pehle naam likho.")
        elif photo is None:
            st.error("Pehle photo lo.")
        else:
            encoding = get_face_encoding_from_image(photo)
            if encoding is None:
                st.error("Face detect nahi hua. Dobara photo lo.")
            else:
                names.append(person_name)
                encodings.append(encoding)
                save_known_faces(names, encodings)
                st.success(f"{person_name} register ho gaya!")

elif menu == "Attendance Lagao":
    st.header("Attendance Lagao")
    if len(names) == 0:
        st.warning("Pehle register karo.")
    else:
        photo = st.camera_input("Apni face ki photo lo")
        if photo is not None:
            encoding = get_face_encoding_from_image(photo)
            if encoding is None:
                st.error("Face detect nahi hua. Dobara photo lo.")
            else:
                matches = face_recognition.compare_faces(encodings, encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(encodings, encoding)
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    matched_name = names[best_match_index]
                    result = mark_attendance(matched_name)
                    st.success(result)
                else:
                    st.error("Ye face register nahi hai.")

elif menu == "Attendance Dekho":
    st.header("Aaj Ki Attendance")
    df = load_today_attendance()
    if df.empty:
        st.info("Aaj abhi tak kisi ki attendance nahi lagi.")
    else:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("CSV Download Karo", csv, "attendance.csv", "text/csv")