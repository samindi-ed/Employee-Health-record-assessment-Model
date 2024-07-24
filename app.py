from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the CBC model
cbc_model = joblib.load('cbc_model.pkl')

# Load the BMP model
bmp_model = joblib.load('bmp_model.pkl')
# Load the other models
lipid_panel_model = joblib.load('lipid_panel_model.pkl')
thyroid_panel_model = joblib.load('thyroid_panel_model.pkl')
cardiac_biomarkers_model = joblib.load('cardiac_biomarkers_model.pkl')
ecg_values_model = joblib.load('ecg_values_model.pkl')

# Define the general features
general_features = ['age', 'height', 'weight', 'BMI']
lipid_features = ['HDL', 'LDL']
thyroid_features = ['T3', 'T4', 'TSH']
cardiac_features = ['hs_cTn', 'BNP', 'NT_proBNP', 'CK', 'CK_MB']
ecg_features = ['RR_interval', 'P_wave', 'PR_interval', 'PR_segment', 'QRS_complex', 'ST_segment', 'T_wave', 'QT_interval']

# Define the features for the CBC section
cbc_features = ['red_blood_cells', 'white_blood_cells', 'platelets', 'hemoglobin', 'hematocrit']
# Define the features for the BMP section
bmp_features = ['BUN', 'creatinine', 'glucose', 'CO2', 'calcium', 'sodium', 'potassium', 'chloride']


class Employee(db.Model):
    __tablename__ = 'employee'

    id = db.Column(db.Integer, primary_key=True)
    # Add more columns as per your employee data
    # Example: name = db.Column(db.String(100), nullable=False)
    # Example: email = db.Column(db.String(100), unique=True, nullable=False)


class EmployeeHealth(db.Model):
    __tablename__ = 'employee_health'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    age = db.Column(db.Float)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    bmi = db.Column(db.Float)
    cbc_rbc = db.Column(db.Float)
    cbc_wbc = db.Column(db.Float)
    cbc_platelets = db.Column(db.Float)
    cbc_hemoglobin = db.Column(db.Float)
    cbc_hematocrit = db.Column(db.Float)
    bmp_bun = db.Column(db.Float)
    bmp_creatinine = db.Column(db.Float)
    bmp_glucose = db.Column(db.Float)
    bmp_co2 = db.Column(db.Float)
    bmp_calcium = db.Column(db.Float)
    bmp_sodium = db.Column(db.Float)
    bmp_potassium = db.Column(db.Float)
    bmp_chloride = db.Column(db.Float)
    lipid_hdl=db.Column(db.Float)
    lipid_ldl=db.Column(db.Float)
    thyroid_t3=db.Column(db.Float)
    thyroid_t4=db.Column(db.Float)
    thyroid_tsh=db.Column(db.Float)
    cardiac_hs_ctn=db.Column(db.Float)
    cardiac_bnp=db.Column(db.Float)
    cardiac_nt_probnp=db.Column(db.Float)
    cardiac_ck=db.Column(db.Float)
    cardiac_ck_mb=db.Column(db.Float)
    ecg_rr_interval=db.Column(db.Float)
    ecg_p_wave=db.Column(db.Float)
    ecg_pr_interval=db.Column(db.Float)
    ecg_pr_segment=db.Column(db.Float)
    ecg_qrs_complex=db.Column(db.Float)
    ecg_st_segment=db.Column(db.Float)
    ecg_t_wave=db.Column(db.Float)
    ecg_qt_interval=db.Column(db.Float)

    # Example: Define relationship to Employee model
    employee = db.relationship('Employee', backref=db.backref('health_records', lazy=True))

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index', methods=['POST'])
def index():
    employee_id = request.form['employee_id']
    session['employee_id'] = employee_id
    mandatory_params = general_features
    sections = {
        "Complete Blood Count": cbc_features,
        "Basic Metabolic Panel": bmp_features,
        "Lipid Panel": lipid_features,
        "Thyroid Panel": thyroid_features,
        "Cardiac Biomarkers": cardiac_features,
        "ECG Values": ecg_features
    }
    return render_template('index.html', employee_id=employee_id, mandatory_params=mandatory_params, sections=sections)

@app.route('/result', methods=['POST'])
def result():
    employee_id = session.get('employee_id')
    all_params = request.form.to_dict()

    def get_float_value(param):
        value = all_params.get(param)
        return float(value) if value else None

    general_data = {param: get_float_value(param) for param in general_features}
    section_results = {}
    recommendations = []

    def process_section(section_name, section_features, model):
        section_data = {param: get_float_value(param) for param in section_features}
        if any(section_data.values()):
            data = {**general_data, **section_data}
            df = pd.DataFrame(data, index=[0])
            missing_cols = set(general_features + section_features) - set(df.columns)
            for col in missing_cols:
                df[col] = 0.0
            df = df[general_features + section_features]
            prediction = model.predict(df)[0]
            if prediction == 1:
                out_of_range_biomarkers = []
                for param, (low, high) in zip(section_features, section_ranges[section_name]):
                    if section_data[param] < low or section_data[param] > high:
                        out_of_range_biomarkers.append(param)
                        recommendations.append(f"Possible issues related to {param}.")
                if out_of_range_biomarkers:
                    recommendations.append(f"Out-of-range biomarkers detected: {', '.join(out_of_range_biomarkers)}.")
                    recommendations.append("It is recommended to consult a doctor for further evaluation.")
                    section_results[section_name] = 'bad'
            else:
                section_results[section_name] = 'good'

    section_ranges = {
        "Complete Blood Count": [(4.5, 6.1), (4.0, 10.8), (150, 400), (13.0, 17.0), (40, 52)],
        "Basic Metabolic Panel": [(6, 20), (0.6, 1.3), (70, 100), (23, 29), (8.5, 10.2), (135, 145), (3.7, 5.2), (96, 106)],
        "Lipid Panel": [(60, float('inf')), (float('-inf'), 100)],
        "Thyroid Panel": [(80, 180), (0.8, 1.8), (0.5, 4)],
        "Cardiac Biomarkers": [(float('-inf'), 1), (float('-inf'), 100), (float('-inf'), 300), (30, 200), (0, 12)],
        "ECG Values": [(0.6, 1.2), (80, 80), (120, 200), (50, 120), (80, 100), (80, 120), (160, 160), (float('-inf'), 420)]
    }

    process_section("Complete Blood Count", cbc_features, cbc_model)
    process_section("Basic Metabolic Panel", bmp_features, bmp_model)
    process_section("Lipid Panel", lipid_features, lipid_panel_model)
    process_section("Thyroid Panel", thyroid_features, thyroid_panel_model)
    process_section("Cardiac Biomarkers", cardiac_features, cardiac_biomarkers_model)
    process_section("ECG Values", ecg_features, ecg_values_model)

    health_status = 'Good' if all(status == 'good' for status in section_results.values()) else 'Bad'
    prediction_text = 'Overall health status is good.' if health_status == 'Good' else 'There are some health issues.'

    health_data = EmployeeHealth(
        employee_id=employee_id,
        age=general_data.get('age'),
        height=general_data.get('height'),
        weight=general_data.get('weight'),
        bmi=general_data.get('BMI'),
        cbc_rbc=all_params.get('red_blood_cells') or None,
        cbc_wbc=all_params.get('white_blood_cells') or None,
        cbc_platelets=all_params.get('platelets') or None,
        cbc_hemoglobin=all_params.get('hemoglobin') or None,
        cbc_hematocrit=all_params.get('hematocrit') or None,
        bmp_bun=all_params.get('bun') or None,
        bmp_creatinine=all_params.get('creatinine') or None,
        bmp_glucose=all_params.get('glucose') or None,
        bmp_co2=all_params.get('co2') or None,
        bmp_calcium=all_params.get('calcium') or None,
        bmp_sodium=all_params.get('sodium') or None,
        bmp_potassium=all_params.get('potassium') or None,
        bmp_chloride=all_params.get('chloride') or None,
        lipid_hdl=all_params.get('HDL') or None,
        lipid_ldl=all_params.get('LDL') or None,
        thyroid_t3=all_params.get('T3') or None,
        thyroid_t4=all_params.get('T4') or None,
        thyroid_tsh=all_params.get('TSH') or None,
        cardiac_hs_ctn=all_params.get('hs_cTn') or None,
        cardiac_bnp=all_params.get('BNP') or None,
        cardiac_nt_probnp=all_params.get('NT_proBNP') or None,
        cardiac_ck=all_params.get('CK') or None,
        cardiac_ck_mb=all_params.get('CK_MB') or None,
        ecg_rr_interval=all_params.get('RR_interval') or None,
        ecg_p_wave=all_params.get('P_wave') or None,
        ecg_pr_interval=all_params.get('PR_interval') or None,
        ecg_pr_segment=all_params.get('PR_segment') or None,
        ecg_qrs_complex=all_params.get('QRS_complex') or None,
        ecg_st_segment=all_params.get('ST_segment') or None,
        ecg_t_wave=all_params.get('T_wave') or None,
        ecg_qt_interval=all_params.get('QT_interval') or None
    )

    db.session.add(health_data)
    db.session.commit()

    return render_template('result.html', prediction_text=prediction_text, prediction_class=health_status.lower(), recommendations=recommendations, section_results=section_results, employee_id=employee_id)

@app.route('/signout', methods=['POST'])
def signout():
    session.pop('employee_id', None)
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
