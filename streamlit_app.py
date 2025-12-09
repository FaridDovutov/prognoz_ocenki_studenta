import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin # Импорт для кастомного класса

# --- 0. ОПРЕДЕЛЕНИЕ ПОЛЬЗОВАТЕЛЬСКОГО ТРАНСФОРМАТОРА (ОБЯЗАТЕЛЬНО!) ---
# Этот класс должен присутствовать в том же файле, где происходит joblib.load
class RangeToMean(BaseEstimator, TransformerMixin):
    """Трансформатор, который преобразует строковые диапазоны и конвертирует запятые в точки."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        for col in X_out.columns:
            def convert_range(value):
                if isinstance(value, str) and '-' in value:
                    try:
                        lower, upper = map(float, value.split('-'))
                        return (lower + upper) / 2
                    except ValueError:
                        return np.nan
                try:
                    if isinstance(value, str):
                        value = value.replace(',', '.')
                    return float(value)
                except (ValueError, TypeError):
                    return np.nan
            
            X_out[col] = X_out[col].apply(convert_range)
        
        X_out = X_out.fillna(X_out.median(numeric_only=True))
        
        return X_out

# --- КОНСТАНТЫ И ЗАГРУЗКА МОДЕЛИ ---

try:
    # Загружаем сохраненный Pipeline, включая RangeToMean и Random Forest
    best_model = joblib.load('student_grade_predictor.pkl')
    MODEL_LOADED = True
except FileNotFoundError:
    st.error("Хатогӣ: Файли модел 'student_grade_predictor.pkl' ёфт нашуд. Лутфан, аввал скрипти омӯзишро иҷро кунед.")
    MODEL_LOADED = False
except Exception as e:
    # Обработка ошибки Attribute Error при загрузке
    st.error(f"Хатогӣ ҳангоми боркунии модел: {e}. Лутфан, санҷед, ки тағйирот дар RangeToMean дар файли омӯзиш ва деплой якхелаанд.")
    MODEL_LOADED = False


# Список признаков, которые ожидает модель (должен совпадать с FEATURES из train_and_save_model.py)
INPUT_FEATURES = ['Student_Age', 'Sex', 'High_School_Type', 'Scholarship', 'A6itional_Work', 
                  'Sports_activity', 'Transportation', 'Weekly_Study_Hours', 'Attendance', 
                  'Reading', 'Notes', 'Listening_in_Class', 'Project_work']

# --- ФУНКЦИИ ПРИЛОЖЕНИЯ ---

def main():
    st.title("🎓 Пешгӯии баҳои имтиҳонотии донишҷӯй (Аз рӯи системаи 100-хола)")
    st.markdown("Барои ба даст овардани баҳои ниҳоӣ додаҳои донишҷӯйро дохил намоед.")
    
    if not MODEL_LOADED:
        return

    # --- Создание форм ввода для пользователя ---
    
    with st.form("student_data_form"):
        # Раздел 1: Демография
        st.header("1. Додаҳои демографӣ ва мактабӣ")
        col1, col2, col3 = st.columns(3)
        
        student_age = col1.number_input("Синну соли донишҷӯй", min_value=15, max_value=30, value=18)
        sex = col2.selectbox("Ҷинс", ["Male", "Female"])
        high_school_type = col3.selectbox("Намуди мактаб", ["Urban", "Rural"])
        
        # Раздел 2: Ресурсы и Активность
        st.header("2. Манбаҳо ва фаъолият")
        col4, col5, col6 = st.columns(3)
        
        scholarship = col4.selectbox("Стипендия", ["Yes", "No"])
        additional_work = col5.selectbox("Ҷойи кори иловагӣ", ["Yes", "No"], key='A6itional_Work')
        sports_activity = col6.selectbox("Варзиш", ["Yes", "No"])

        transportation = st.selectbox("Воситаи нақлиёт (Автобус, шахсӣ ва ғайра)", ["Bus", "Private", "Other"])
        weekly_study_hours = st.text_input("Соатҳои таълимии ҳафтаина (масалан, 10-15 ё 20)", value="10-15")
        attendance = st.slider("Фоизи иштирок дар дарс (%)", min_value=0, max_value=100, value=90)
        
        # Раздел 3: Учебное поведение
        st.header("3. Фаъолияти таълимӣ")
        
        reading = st.selectbox("Хондани адабиёти иловагӣ", ["Yes", "No"])
        notes = st.selectbox("Навиштани матни лексияҳо", ["Yes", "No"])
        listening_in_class = st.selectbox("Дар дарсҳо бодиққат аст", ["Yes", "No"])
        project_work = st.selectbox("Сифати корҳои супоришӣ", ["Good", "Average", "Poor"])
        
        # Кнопка отправки формы
        submitted = st.form_submit_button("Пешгӯии баҳо")

    # --- Логика прогнозирования ---
    if submitted:
        # ИСПРАВЛЕНИЕ ОШИБКИ: ЯВНОЕ ПРЕОБРАЗОВАНИЕ ЧИСЛОВЫХ ДАННЫХ В float 
        input_data = pd.DataFrame({
            'Student_Age': [float(student_age)],  
            'Sex': [sex],
            'High_School_Type': [high_school_type],
            'Scholarship': [scholarship],
            'A6itional_Work': [additional_work],
            'Sports_activity': [sports_activity],
            'Transportation': [transportation],
            'Weekly_Study_Hours': [weekly_study_hours], # Оставляем строкой, RangeToMean его обработает
            'Attendance': [float(attendance)],    
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
            
            predicted_score = round(prediction, 2)

            st.success(f"### 🎯 Пешгӯии баҳои имтиҳонотии ниҳоӣ (аз 100-хол):")
            st.success(f"## {predicted_score}")
            
            st.info(f"Ин баҳо ба {round(predicted_score/10, 2)} аз рӯи системаи 10-хола баробар аст.")
            
        except Exception as e:
            st.error(f"Ҳангоми пешгӯикунӣ хатогӣ пайдо шуд: {e}")
            st.warning("Санҷед, ки оё намуди додаҳои дохилкардашуда дуруст аст. Ҳамаи майдонҳои рақамӣ бояд дар формати ададӣ бошанд.")


if __name__ == '__main__':
    main()
