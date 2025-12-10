import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin 
# Дополнительный импорт для совместимости pickle/joblib с Python 3.13+

# --- 0. ОПРЕДЕЛЕНИЕ ПОЛЬЗОВАТЕЛЬСКОГО ТРАНСФОРМАТОРА (Критически важно!) ---
# Класс должен быть идентичен классу в файле обучения.
class RangeToMean(BaseEstimator, TransformerMixin):
    """Трансформатор, который преобразует строковые диапазоны и конвертирует запятые в точки."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        for col in X_out.columns:
            def convert_range(value):
                # Если значение - строка, удаляем лишние пробелы для безопасности
                if isinstance(value, str):
                    value = value.strip()
                    # Шаг 1: Обработка диапазонов типа '19-22'
                    if '-' in value:
                        try:
                            parts = value.replace(',', '.').split('-')
                            lower, upper = map(float, parts)
                            return (lower + upper) / 2
                        except ValueError:
                            return np.nan
                
                # Шаг 2: Конвертация простых чисел и строк с запятыми в float
                try:
                    if isinstance(value, str):
                        value = value.replace(',', '.')
                    return float(value)
                except (ValueError, TypeError):
                    return np.nan

            X_out[col] = X_out[col].apply(convert_range)
        
        X_out = X_out.fillna(X_out.median(numeric_only=True))
        
        return X_out

# --- СЛОВАРИ СОПОСТАВЛЕНИЙ (MAPPING) ---
# Сопоставляем отображаемый текст с ожидаемыми латинскими значениями модели
MAPPING = {
    'Мужской / Male': 'Male',
    'Женский / Female': 'Female',
    'Городская / Urban': 'Urban',
    'Сельская / Rural': 'Rural',
    'Да / Yes': 'Yes',
    'Нет / No': 'No',
    'Хорошо / Good': 'Good',
    'Средне / Average': 'Average',
    'Плохо / Poor': 'Poor',
    'Автобус / Bus': 'Bus',
    'Личный / Private': 'Private',
    'Другое / Other': 'Other'
}


# --- КОНСТАНТЫ И ЗАГРУЗКА МОДЕЛИ ---

try:
    best_model = joblib.load('student_grade_predictor.pkl')
    MODEL_LOADED = True
except FileNotFoundError:
    st.error("Хатогӣ: Файли модел 'student_grade_predictor.pkl' ёфт нашуд. Лутфан, аввал скрипти омӯзишро иҷро кунед.")
    MODEL_LOADED = False
except Exception as e:
    st.error(f"Хатогӣ ҳангоми боркунии модел: {e}")
    MODEL_LOADED = False


INPUT_FEATURES = ['Student_Age', 'Sex', 'High_School_Type', 'Scholarship', 'A6itional_Work', 
                  'Sports_activity', 'Transportation', 'Weekly_Study_Hours', 'Attendance', 
                  'Reading', 'Notes', 'Listening_in_Class', 'Project_work']

# --- ОСНОВНАЯ ФУНКЦИЯ ПРИЛОЖЕНИЯ ---

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
        sex_display = col2.selectbox("Ҷинс", ["Мужской / Male", "Женский / Female"])
        high_school_type_display = col3.selectbox("Намуди мактаб", ["Городская / Urban", "Сельская / Rural"])
        
        # Раздел 2: Ресурсы и Активность
        st.header("2. Манбаҳо ва фаъолият")
        col4, col5, col6 = st.columns(3)
        
        scholarship_display = col4.selectbox("Стипендия", ["Да / Yes", "Нет / No"])
        additional_work_display = col5.selectbox("Ҷойи кори иловагӣ", ["Да / Yes", "Нет / No"], key='A6itional_Work')
        sports_activity_display = col6.selectbox("Варзиш", ["Да / Yes", "Нет / No"])

        transportation_display = st.selectbox("Воситаи нақлиёт", ["Автобус / Bus", "Личный / Private", "Другое / Other"])
        weekly_study_hours = st.text_input("Соатҳои таълимии ҳафтаина (масалан, 10-15 ё 20)", value="10-15")
        attendance = st.slider("Фоизи иштирок дар дарс (%)", min_value=0, max_value=100, value=90)
        
        # Раздел 3: Учебное поведение
        st.header("3. Фаъолияти таълимӣ")
        
        reading_display = st.selectbox("Хондани адабиёти иловагӣ", ["Да / Yes", "Нет / No"])
        notes_display = st.selectbox("Навиштани матни лексияҳо", ["Да / Yes", "Нет / No"])
        listening_in_class_display = st.selectbox("Дар дарсҳо бодиққат аст", ["Да / Yes", "Нет / No"])
        project_work_display = st.selectbox("Сифати корҳои супоришӣ", ["Хорошо / Good", "Средне / Average", "Плохо / Poor"])
        
        submitted = st.form_submit_button("Пешгӯии баҳо")

    # --- Логика прогнозирования ---
    if submitted:
        
        # ПРЕОБРАЗОВАНИЕ ВВОДА В ФОРМАТ, ОЖИДАЕМЫЙ МОДЕЛЬЮ
        input_data = pd.DataFrame({
            # Числовые (ЯВНО float для предотвращения ошибки 'ufunc isnan')
            'Student_Age': [float(student_age)],
            'Attendance': [float(attendance)],
            'Weekly_Study_Hours': [weekly_study_hours], # Оставляем строкой для RangeToMean
            
            # Категориальные (Используем словарь MAPPING)
            'Sex': [MAPPING[sex_display]],
            'High_School_Type': [MAPPING[high_school_type_display]],
            'Scholarship': [MAPPING[scholarship_display]],
            'A6itional_Work': [MAPPING[additional_work_display]],
            'Sports_activity': [MAPPING[sports_activity_display]],
            'Transportation': [MAPPING[transportation_display]],
            'Reading': [MAPPING[reading_display]],
            'Notes': [MAPPING[notes_display]],
            'Listening_in_Class': [MAPPING[listening_in_class_display]],
            'Project_work': [MAPPING[project_work_display]]
        })
        
        # Гарантируем правильный порядок столбцов
        input_data = input_data[INPUT_FEATURES]

        try:
            prediction = best_model.predict(input_data)[0]
            predicted_score = round(prediction, 2)

            st.success(f"### 🎯 Пешгӯии баҳои имтиҳонотии ниҳоӣ (аз 100-хол):")
            st.success(f"## {predicted_score}")
            
            st.info(f"Ин баҳо ба {round(predicted_score/10, 2)} аз рӯи системаи 10-хола баробар аст.")
            
        except Exception as e:
            st.error(f"Ҳангоми пешгӯикунӣ хатогӣ пайдо шуд: {e}")
            st.warning("Санҷед, ки ҳамаи поляҳо пур карда шудаанд ва 'Соатҳои таълимии ҳафтаина' дар правильном формате (например, 10-15 или 20).")


if __name__ == '__main__':
    main()
