import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- КОНСТАНТЫ И ЗАГРУЗКА МОДЕЛИ ---
# Загружаем сохраненную модель
try:
    best_model = joblib.load('student_grade_predictor.pkl')
    MODEL_LOADED = True
except FileNotFoundError:
    st.error("Ошибка: Файл модели 'student_grade_predictor.pkl' не найден. Сначала запустите train_and_save_model.py!")
    MODEL_LOADED = False

# Список признаков, которые ожидает модель (должен совпадать с FEATURES из train_and_save_model.py)
INPUT_FEATURES = ['Student_Age', 'Sex', 'High_School_Type', 'Scholarship', 'A6itional_Work', 
                  'Sports_activity', 'Transportation', 'Weekly_Study_Hours', 'Attendance', 
                  'Reading', 'Notes', 'Listening_in_Class', 'Project_work']

# --- ФУНКЦИИ ПРИЛОЖЕНИЯ ---

def main():
    st.title("🎓 Прогнозирование оценки студента (100-балльная система)")
    st.markdown("Введите данные студента, чтобы получить прогноз итоговой оценки.")
    
    if not MODEL_LOADED:
        return

    # --- Создание форм ввода для пользователя ---
    
    with st.form("student_data_form"):
        # Раздел 1: Демография
        st.header("1. Демографические данные и Школа")
        col1, col2, col3 = st.columns(3)
        
        student_age = col1.number_input("Возраст студента", min_value=15, max_value=30, value=18)
        sex = col2.selectbox("Пол", ["Male", "Female"])
        high_school_type = col3.selectbox("Тип школы", ["Urban", "Rural"])
        
        # Раздел 2: Ресурсы и Активность
        st.header("2. Ресурсы и Занятость")
        col4, col5, col6 = st.columns(3)
        
        scholarship = col4.selectbox("Стипендия", ["Yes", "No"])
        additional_work = col5.selectbox("Дополнительная работа", ["Yes", "No"], key='A6itional_Work')
        sports_activity = col6.selectbox("Спорт", ["Yes", "No"])

        transportation = st.selectbox("Транспорт (Автобус, Личный, и т.д.)", ["Bus", "Private", "Other"])
        weekly_study_hours = st.text_input("Еженедельные часы учебы (например, 10-15 или 20)", value="10-15")
        attendance = st.slider("Процент посещаемости (%)", min_value=0, max_value=100, value=90)
        
        # Раздел 3: Учебное поведение
        st.header("3. Учебное поведение")
        
        reading = st.selectbox("Чтение дополнительной литературы", ["Yes", "No"])
        notes = st.selectbox("Ведение конспектов", ["Yes", "No"])
        listening_in_class = st.selectbox("Активное слушание в классе", ["Yes", "No"])
        project_work = st.selectbox("Качество проектных работ", ["Good", "Average", "Poor"])
        
        # Кнопка отправки формы
        submitted = st.form_submit_button("Спрогнозировать оценку")

    # --- Логика прогнозирования ---
    if submitted:
        # Создаем DataFrame из пользовательского ввода
        input_data = pd.DataFrame({
            'Student_Age': [student_age],
            'Sex': [sex],
            'High_School_Type': [high_school_type],
            'Scholarship': [scholarship],
            'A6itional_Work': [additional_work],
            'Sports_activity': [sports_activity],
            'Transportation': [transportation],
            'Weekly_Study_Hours': [weekly_study_hours],
            'Attendance': [attendance],
            'Reading': [reading],
            'Notes': [notes],
            'Listening_in_Class': [listening_in_class],
            'Project_work': [project_work]
        })
        
        # Гарантируем, что порядок столбцов совпадает с порядком в модели
        input_data = input_data[INPUT_FEATURES]

        try:
            # Делаем прогноз с помощью загруженной модели
            prediction = best_model.predict(input_data)[0]
            
            # Округление до двух знаков после запятой
            predicted_score = round(prediction, 2)

            st.success(f"### 🎯 Прогноз итоговой оценки (100-балльная шкала):")
            st.success(f"## {predicted_score}")
            
            st.info(f"Напоминание: Это число соответствует оценке {round(predicted_score/10, 2)} по 10-балльной системе.")
            
        except Exception as e:
            st.error(f"Произошла ошибка при прогнозировании: {e}")
            st.warning("Проверьте, что формат введенных данных соответствует ожидаемому.")


if __name__ == '__main__':
    main()
