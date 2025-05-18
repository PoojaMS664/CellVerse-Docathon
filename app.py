import streamlit as st # type: ignore
import base64
from PIL import Image # type: ignore
import pandas as pd # type: ignore
from utils.nutrition import get_personalized_diet
from utils.exercise import get_personalized_exercise
from utils.screening import calculate_bmi, interpret_blood_sugar, lifestyle_risk, recommend_water_intake
from utils.bot import ask_cohere_bot, ask_faq_bot
from utils.prediction import predict_diabetes_ml
import cohere

co = cohere.Client("PVibIuSTwdGlHJWOV3cLFy0iCV9lzqHx9B4lp8eG")  # Replace with your actual Cohere key

def get_health_tips(age, bmi, risk):
    prompt = f"""
You are a health advisor. Provide exactly 3 short and practical daily lifestyle tips for a person who is {age} years old with a BMI of {bmi} and a diabetes risk level of {risk}.

Format:
1. ....
2. ...
3. ...
"""
    response = co.chat(
        model='command-r',
        message=prompt,
        temperature=0.7
    )
    return response.text.strip()

def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

co = cohere.Client("PVibIuSTwdGlHJWOV3cLFy0iCV9lzqHx9B4lp8eG")  # Replace with your actual Cohere key

# Load dataset
df = pd.read_csv("Final_Diabetes_Prediction_Dataset_with_Videos.csv")

# Configure Streamlit
st.set_page_config(page_title="DiaPredict", layout="wide")

# Inject custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Top navigation bar (unchanged)
logo_base64 = get_base64_image("static/logo.jpg")
st.markdown(f"""
    <style>
        .nav-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(to right, #2C3539, #4F4F4F); /* Gunmetal to Gray */
            padding: 1.2rem 2rem;
            box-shadow: 0 2px 10px rgba(255,255,255,0.1);
            font-family: 'Segoe UI', sans-serif;
            flex-wrap: wrap;
        }}
        .nav-left {{
            display: flex;
            align-items: center;
            gap: 2rem;
        }}
        .logo-img {{
            height: 110px;
            width: 120px;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 4px 12px black;
        }}
        .logo-title {{
            position: absolute;
            left: 53%;
            transform: translateX(-50%);
            color: white;
            font-size: 6rem;
            font-weight: 900;
            letter-spacing: 1.5px;
            white-space: nowrap;
            text-shadow: 6px 6px 10px black;
        }}
        .nav-links {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 0.5rem;
        }}
        .nav-links a {{
            font-size: 1rem;
            text-decoration: none;
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .nav-links a:hover {{
            color: #66ccff;
        }}
        @media screen and (max-width:1000px) {{
            .logo-img {{ height:80px; }}
            .logo-title {{ font-size:1.5rem; }}
            .nav-links a {{ font-size:0.95rem; }}
        }}
    </style>
    <div class="nav-container">
        <div class="nav-left">
            <img src="data:image/png;base64,{logo_base64}" class="logo-img">
            <div class="logo-title">DIAPREDICT</div>
        </div>
        <div class="nav-links">
            <a href="/?page=Home" target="_self">Home</a>
            <a href="/?page=Screening%20Tools" target="_self">Screening Tools</a>
            <a href="/?page=Prediction" target="_self">Prediction</a>
            <a href="/?page=Diet%20%26%20Nutrition" target="_self">Diet & Nutrition</a>
            <a href="/?page=Physical%20Activity" target="_self">Physical Activity</a>
            <a href="/?page=Diagnostic%20Tests" target="_self">Diagnostic Tests</a>
            <a href="/?page=Ask%20the%20Bot" target="_self">Ask the Bot</a>
            <a href="/?page=Patient%20Report" target="_self">Patient Report</a>
            <a href="/?page=Emergency%20Guide" target="_self">Emergency Guide</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# 1️⃣ Read the page query param, default to "Landing"
params = st.query_params
page = params.get("page", "Landing")

if "report" not in st.session_state:
    st.session_state.report = {}

# 2️⃣ If we're on Landing, show only the hero, then stop
background_base64 = get_base64_image("static/background.jpg")
if page == "Landing":
    st.markdown(f"""
        <style>
            .hero-section {{
                position: relative;
                height: 90vh;
                background-image: url("data:image/jpg;base64,{background_base64}");
                background-size: cover;
                background-position: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                color: white;
                text-align: center;
            }}
            .hero-section h1 {{ font-size: 4.5rem; text-shadow: 3px 3px 8px black; }}
            .hero-section p {{
                font-size: 2.4rem;
                color: white;
            }}

        </style>
        <div class="hero-section">
            <h1>Welcome to DiaPredict</h1>
            <p>Know Your Risk, Take Control</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
         encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}"); # type: ignore # type: ignore # type: ignore
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def add_bg_to_main_content(image_file):
    import base64
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 3️⃣ From here on, page is Home, Prediction, etc.

# ----------------------- Pages -----------------------
if page == "Home":
    st.title("🏠 Welcome to DiaPredict – Know Your Risk, Take Control")

    # --- What is Diabetes Section ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("## 🩺 What is Diabetes?")
        st.markdown("""
        Diabetes is a chronic health condition in which the level of glucose (sugar) in your blood becomes too high.It occurs when your pancreas doesn’t produce enough insulin — a hormone that helps glucose enter your cells for energy — or when your body’s cells become resistant to insulin’s effects.Glucose, which primarily comes from carbohydrates in food, serves as the main source of energy. Insulin acts like a key, allowing glucose to move from your bloodstream into your cells.
        
        When insulin is insufficient or ineffective, glucose builds up in your blood, leading to hyperglycemia.This buildup can silently damage organs over time. Uncontrolled diabetes can lead to complications like cardiovascular disease, neuropathy, kidney failure, poor wound healing, and vision loss.Diabetes can affect anyone, regardless of age or lifestyle. While it's lifelong, it can be managed with diet, exercise, monitoring, and medications.
        """)
    with col2:
        st.image("static/Diabetes.jpg")

    st.markdown("---")
    
    st.markdown("## 🔍 Types of Diabetes")
    # Type 1 Diabetes
    st.markdown("## 💉 Type 1 Diabetes")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("static/type1.jpg", caption="Symptoms of Type 1 Diabetes")
    with col2:
        st.markdown("""
        *Type 1 diabetes* is a chronic condition in which the pancreas produces *little or no insulin*.  
        It's often diagnosed in children and young adults.

        Common symptoms include:  
        - Increased thirst and frequent urination 💧  
        - Extreme hunger 🍽  
        - Unintended weight loss ⚖  
        - Fatigue 😴  
        - Blurred vision 👁  
        - Mood changes 😠

        Insulin is an important hormone that regulates the amount of glucose (sugar) in your blood. Under normal circumstances, insulin functions in the following steps:

        Your body breaks down the food you eat into glucose (sugar), which is your body’s main source of energy.
        Glucose enters your bloodstream, which signals your pancreas to release insulin.
        Insulin helps glucose in your blood enter your muscle, fat and liver cells so they can use it for energy or store it for later use.
        When glucose enters your cells and the levels in your bloodstream decrease, it signals your pancreas to stop producing insulin.
        If you don’t have enough insulin, too much sugar builds up in your blood, causing hyperglycemia (high blood sugar), and your body can’t use the food you eat for energy. This can lead to serious health problems or even death if it’s not treated. People with Type 1 diabetes need synthetic insulin every day in order to live and be healthy.

        Type 1 diabetes was previously known as juvenile diabetes and insulin-dependent diabetes.
        """)
    st.markdown("---")

    # Type 2 Diabetes
    st.markdown("## 💊 Type 2 Diabetes")
    col3, col4 = st.columns([2, 1])
    with col3:
        st.markdown("""
        Type 2 diabetes (T2D) is a chronic condition that happens when you have persistently high blood sugar levels (hyperglycemia).
    
        Symptoms:  
        - Significant weight loss ⚖
        - Increased water intake 💧
        - Increased frequency of urination 🚽
        - Tiredness 😴  

       Healthy blood sugar (glucose) levels are 70 to 99 milligrams per deciliter (mg/dL). If you have undiagnosed Type 2 diabetes, your levels are typically 126 mg/dL or higher.

        T2D happens because your pancreas doesn’t make enough insulin (a hormone), your body doesn’t use insulin properly, or both. This is different from Type 1 diabetes, which happens when an autoimmune attack on your pancreas results in a total lack of insulin production.  
        Type 2 diabetes is very common. More than 37 million people in the U.S. have diabetes (about 1 in 10 people), and about 90% to 95% of them have T2D.

        Researchers estimate that T2D affects about 6.3% of the world’s population. T2D most commonly affects adults over 45, but people younger than 45 can have it as well, including children.
        """)
    with col4:
        st.image("static/type2.jpg", caption="Type 2 Diabetes Awareness")
    st.markdown("---")

    # Gestational Diabetes
    st.markdown("## 🤰  Gestational Diabetes")
    col5, col6 = st.columns([1, 2])
    with col5:
        st.image("static/gest.jpg", caption="Risk Factors & Symptoms of Gestational Diabetes")
    with col6:
        st.markdown("""
        Gestational diabetes (GD or GDM) is a type of diabetes that develops exclusively in pregnancy when blood sugar levels get too high (hyperglycemia). It happens when the hormones from the placenta block your ability to use or make insulin. Insulin helps your body maintain the right amount of glucose in your blood. Too much glucose in your blood can lead to pregnancy complications. GD usually appears during the middle of pregnancy, between 24 and 28 weeks. Your pregnancy care provider will order a blood test to check for gestational diabetes.

        Common risk factors include:  
        - Obesity or excessive weight before pregnancy ⚖  
        - Age over 35 👩‍🦳  
        - Personal or family history of diabetes 🧬  
        - Polycystic ovary syndrome (PCOS) 🧪

       Developing GD doesn’t mean you already had diabetes before you got pregnant. The condition appears because of pregnancy. People with Type 1 and Type 2 diabetes before pregnancy have their own, separate challenges when they become pregnant.

        Fortunately, gestational diabetes is well understood, and healthcare providers are usually able to help you manage the condition with small lifestyle and dietary changes. Most people don’t experience serious complications from gestational diabetes and deliver healthy babies.
        
        Prevention of Gestational Diabetes:
        While gestational diabetes cannot always be prevented, adopting a healthy lifestyle before and during pregnancy can significantly reduce the risk. Maintaining a balanced diet rich in whole grains, fruits, vegetables, and lean proteins helps regulate blood sugar levels. Regular physical activity, such as walking or prenatal exercises, improves insulin sensitivity. Achieving and maintaining a healthy weight before pregnancy and managing weight gain during pregnancy are also key. If you have risk factors, early screening and regular prenatal checkups can help detect and manage blood sugar changes promptly.
       
       Why Gestational Diabetes Often Has No Warning Signs

        - GD usually develops gradually without obvious symptoms.

        - Common pregnancy symptoms (fatigue, thirst, frequent urination) overlap with GD signs.

        - Mild blood sugar elevations often go unnoticed.
        """)
    st.markdown("---")

    # Prediabetes
    st.markdown("## ⚠ Prediabetes")
    col7, col8 = st.columns([2, 1])
    with col7:
        st.markdown("""
        Prediabetes happens when you have elevated blood sugar levels, but they’re not high enough to be considered Type 2 diabetes.

        - ✅ *Healthy range:* 70–99 mg/dL (FBS)
        - ⚠ *Prediabetic range:* 100–125 mg/dL (FBS)

        According to the *American Diabetes Association, for a 45-year-old with prediabetes, the **10-year risk* of developing Type 2 diabetes is *9% to 14%*.

        Luckily, prediabetes is *reversible* with:
        - Healthy eating 🥗  
        - Regular exercise 🏃  
        - Weight loss ⚖  
        - Glucose monitoring 💉
        """)
    with col8:
        st.image("static/prediabete.jpg", caption="Managing Prediabetes with Lifestyle")

    st.markdown("---")
    
    st.markdown("## 🔴 Complications of Uncontrolled Diabetes")
    col9, col10 = st.columns([1, 2])
    with col9:
        st.image("static/complication.jpg", caption="Common Complications of Diabetes")
    with col10:
        st.markdown("""
        ###  Cardiovascular Disease  
        High blood sugar damages arteries, increasing the risk of heart attack, stroke, and high blood pressure.

        ### 🧠 Nerve Damage (Diabetic Neuropathy)  
        Causes tingling, pain, numbness—especially in hands and feet. May lead to infections, ulcers, and amputations.

        ### 👁 Eye Damage (Diabetic Retinopathy)  
        Affects retina blood vessels; may result in blurred vision, vision loss, or blindness.

        ### 🧪 Kidney Damage (Diabetic Nephropathy)  
        Slowly impairs kidney filtering function; can lead to chronic kidney disease or kidney failure.

        ### 🧬 Weakened Immune System  
        Increases vulnerability to infections and delays wound healing.

        ### ⚠ Foot Complications  
        Reduced blood flow and nerve sensitivity increase risk of ulcers, gangrene, and amputations.
        """)
    st.markdown("---")
    
    # --- Why Early Screening Matters ---
    st.markdown("## 🧪 Why Early Screening Matters")
    col3, col4 = st.columns([3, 2])
    with col3:
        st.markdown("""
        Early screening for diabetes is crucial because it allows us to detect high blood sugar levels or insulin resistance before symptoms appear. Many people with prediabetes or even Type 2 diabetes are unaware they have it, as early stages often show no clear symptoms. By identifying the condition early:

    🧬 Biological damage can be prevented: High blood glucose levels silently damage blood vessels, nerves, kidneys, and eyes over time. Screening helps stop this before it starts.

    ⏳ Timely lifestyle changes: With early detection, simple changes in diet, physical activity, and weight management can reduce the risk of developing Type 2 diabetes by up to 70%.

    💰 Lower healthcare costs: Early management avoids expensive long-term treatments for complications like heart disease, kidney failure, or amputations.

    🩺 Better health tracking: Screening enables ongoing monitoring of key indicators like BMI, blood sugar, and hydration, helping users stay in control.

    🧠 Empowerment through awareness: Knowing your risk motivates healthier decisions and gives people the tools to protect their future.
        """)
    with col4:
        st.image("static/matters.jpg")

    st.markdown("---")
    
    st.markdown("### ✅ Take charge of your health with DiaPredict!")
    st.success("Use the top navigation to explore prediction tools, diet plans, exercise guidance, tests, chatbot, and reports.")


elif page == "Screening Tools":
    st.header("🧪 Screening Tools")
    img_base = get_base64_image("static\Screening Tools.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_base}");
            background-attachment: fixed;
            background-size: cover;
        }}

        h1, h2, h3, h4, h5, h6, p, label, .css-10trblm, .css-1d391kg, .css-1cpxqw2 {{
            color: black !important;
            font-size: 24px !important;
            font-weight: 600;
        }}

        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stTextArea>div>textarea {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: black !important;
            font-size: 18px !important;
        }}

        .stButton>button {{
            font-size: 18px !important;
            font-weight: bold;
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # BMI Calculator
    st.subheader("1. 📏 BMI Calculator")
    weight = st.number_input("Weight (kg)", 30.0, 200.0)
    height = st.number_input("Height (m)", 1.0, 2.5)
    if st.button("Calculate BMI"):
        bmi, status = calculate_bmi(weight, height)
        st.success(f"Your BMI is *{bmi}* — {status}")

    # Random Blood Sugar Interpretation
    st.subheader("2. 🩸 Random Blood Sugar Interpretation")
    sugar = st.number_input("Blood Sugar (mg/dL)", 60.0, 500.0)
    if sugar:
        interpretation = interpret_blood_sugar(sugar)
        st.info(f"Blood Sugar Interpretation: {interpretation}")

    # Water Intake Recommendation
    st.subheader("3. 💧 Water Intake Recommendation")
    age = st.number_input("Age", 1, 120)
    water = st.number_input("Water Intake (L)", 0.0, 10.0, step=0.1)
    if age and water:
        recommended = recommend_water_intake(weight) # type: ignore
        st.write(f"Recommended Water Intake: *{recommended} L/day*")
        if water < recommended - 0.5:
            st.info("💧 You may need to increase your intake.")
        elif water > recommended + 0.5:
            st.info("🚱 Too much water is not necessary.")
        else:
            st.success("✅ Your water intake looks good!")

    # Lifestyle Risk
    st.subheader("4. ⚠ Lifestyle Risk Evaluation")
    smoke = st.selectbox("Do you smoke?", ["Yes", "No"])
    alcohol = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
    inactive = st.selectbox("Exercise < 4x/week?", ["Yes", "No"])
    if st.button("Evaluate Lifestyle Risk"):
        result = lifestyle_risk(smoke, alcohol, inactive)
        st.warning(result)

# ---- Prediction Page ----
elif page == "Prediction":

    st.markdown("<h2 style='font-size:36px;'>🔮 Diabetes Risk Prediction</h2>", unsafe_allow_html=True)
    st.markdown("Please fill out the form below to assess your diabetes risk.")
    prediction_img = get_base64_image("static/prediction.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{prediction_img}");
            background-attachment: fixed;
            background-size: cover;
        }}
        h1, h2, h3, h4, h5, h6, p, label, .css-10trblm, .css-1d391kg, .css-1cpxqw2 {{
            color: black !important;
            font-size: 24px !important;
            font-weight: 600;
        }}
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stTextArea>div>textarea {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: black !important;
            font-size: 18px !important;
        }}
        .stButton>button {{
            font-size: 18px !important;
            font-weight: bold;
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    def styled_label(text):
        return f"""
        <div style='
            font-size:22px;
            font-weight:700;
            color:black;
            margin-bottom:6px;
        '>{text}</div>
        """

    st.markdown("<h4 style='font-size:28px; font-weight:700;'>🧾 Personal & Lifestyle Information</h4>", unsafe_allow_html=True)

    st.markdown(styled_label("📏 BMI"), unsafe_allow_html=True)
    bmi = st.number_input("", min_value=10.0, max_value=60.0, key="bmi")

    st.markdown(styled_label("🎂 Age"), unsafe_allow_html=True)
    age = st.number_input("", min_value=1, max_value=120, key="age")

    st.markdown(styled_label("🩺 Blood Pressure Level"), unsafe_allow_html=True)
    bp = st.selectbox("", ["Low", "High"], key="bp")

    st.markdown(styled_label("⚖ Recent Weight Change?"), unsafe_allow_html=True)
    weight_change = st.selectbox("", ["No", "Weight Gain", "Weight Loss"], key="weight_change")

    st.markdown(styled_label("🍷 Alcoholic Habits"), unsafe_allow_html=True)
    alcohol = st.selectbox("", ["Yes", "No"], key="alcohol")

    st.markdown(styled_label("🚬 Smoking"), unsafe_allow_html=True)
    smoking = st.selectbox("", ["Yes", "No"], key="smoking")

    st.markdown(styled_label("🚽 Frequent Urination?"), unsafe_allow_html=True)
    urinating = st.selectbox("", ["Yes", "No"], key="urinating")

    st.markdown(styled_label("💧 Increased Thirst?"), unsafe_allow_html=True)
    thirst = st.selectbox("", ["Yes", "No"], key="thirst")

    st.divider()

    st.markdown("<h4 style='font-size:28px; font-weight:700;'>👨‍👩‍👧 Family & Medical History</h4>", unsafe_allow_html=True)

    st.markdown(styled_label("🧬 Family History"), unsafe_allow_html=True)
    genetics = st.selectbox("", ["No", "Mother", "Father", "Both Parents"], key="genetics")

    st.markdown(styled_label("💊 Any Diseases or Medications?"), unsafe_allow_html=True)
    diseases = st.selectbox("", ["Yes", "No"], key="diseases")

    # Initialize report session keys if not already present
    if "report" not in st.session_state:
        st.session_state.report = {}
    if "risk" not in st.session_state:
        st.session_state["risk"] = None
    if "tips" not in st.session_state:
        st.session_state["tips"] = None

    # Step 1: Predict button
    if st.button("🧠 Predict Diabetes Risk"):
        data = {
            "BMI": bmi,
            "Age": age,
            "Blood Pressure Level": bp,
            "Weight Change": weight_change,
            "Alcoholic Habits": alcohol,
            "Smoking": smoking,
            "Frequent Urination": urinating,
            "Increased Thirst": thirst,
            "Family History": genetics,
            "Diseases": diseases,
        }
        prediction = predict_diabetes_ml(data)
        risk = "High" if prediction == 1 else "Low"
        advice = "⚠ High risk: Please consult a doctor and follow a healthy routine." if risk == "High" else "✅ Low risk: Maintain your healthy lifestyle."

        # Save prediction data to report
        st.session_state.report.update({
            "Age": age,
            "BMI": bmi,
            "Weight Change": weight_change,
            "Diabetes Risk": risk,
            "Prediction Advice": advice
        })
        st.session_state["risk"] = risk

        st.success(f"🧠 Your Diabetes Risk is: {risk}")
        st.info(advice)

    # Step 2: Get AI tips (only if prediction done)
    if st.session_state.get("risk"):
        if st.button("📌 Get AI Health Tips"):
            with st.spinner("Generating personalized tips..."):
                try:
                    risk = st.session_state["risk"]
                    tips = get_health_tips(age, bmi, risk)
                    st.markdown("### 🧠 AI-Powered Lifestyle Tips")
                    st.markdown(tips)
                    st.session_state["tips"] = tips
                    st.session_state.report["Health Tips"] = tips
                except Exception as e:
                    st.error("❌ Failed to generate tips.")
                    st.exception(e)

    # Step 3: Show Download Button (only if both prediction and tips available)
    if st.session_state.get("risk") and st.session_state.get("tips"):
        report = st.session_state.report
        pred_text = f"""DIABETES PREDICTION REPORT
Age: {report.get("Age")}
BMI: {report.get("BMI")}
Weight Change: {report.get("Weight Change")}
Risk: {report.get("Diabetes Risk")}
Advice: {report.get("Prediction Advice")}

🧠 AI-Powered Lifestyle Tips:
{report.get("Health Tips")}
"""
        st.download_button("📄 Download Prediction Report", pred_text, "prediction_report.txt", "text/plain")
        
elif page == "Diet & Nutrition":
    diet_img = get_base64_image("static/diet.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{diet_img}");
            background-attachment: fixed;
            background-size: cover;
        }}

        h1, h2, h3, h4, h5, h6, p, label, .css-10trblm, .css-1d391kg, .css-1cpxqw2 {{
            color: black !important;
            font-size: 24px !important;
            font-weight: 600;
        }}

        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stTextArea>div>textarea {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: black !important;
            font-size: 18px !important;
        }}

        .stButton>button {{
            font-size: 18px !important;
            font-weight: bold;
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("🥗 Diet & Nutrition Plan")

    risk = st.selectbox("Diabetes Risk", ["High", "Low"])
    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", 30.0, 200.0)
    category = st.selectbox("Nutrition Category", df["Nutrition Category"].dropna().unique())
    sugar = st.number_input("Blood Sugar Level (mg/dL)", 60.0, 500.0)

    match = df[
        (df["Diabetes Risk"].str.lower() == risk.lower()) &
        (df["Gender"].str.lower() == gender.lower()) &
        (df["Nutrition Category"].str.lower() == category.lower())
    ]

    if not match.empty:
        person = match.iloc[0]
        result = get_personalized_diet(person)

        st.session_state.report.update({
            "Diet Plan": result["FoodPlan"],
            "Energy": result["Energy"],
            "Protein": result["Protein"]
        })

        # Display report section
        st.markdown("### 🥗 Diet Summary")

# Energy and Protein with emoji and bold styling
        st.markdown(f"🔋 <b>Energy Requirement:</b><br>{result['Energy']} kcal/day", unsafe_allow_html=True)
        st.markdown(f"💪 <b>Protein Requirement:</b><br>{result['Protein']} g/day", unsafe_allow_html=True)


        st.markdown("🍽 *Food Plan:*", unsafe_allow_html=True)
        st.markdown(
    f"<p style='color:black; font-size:18px;'>{result['FoodPlan']}</p>",
    unsafe_allow_html=True
)

        diet_text = f"""DIET & NUTRITION REPORT
Energy: {result['Energy']} kcal/day
Protein: {result['Protein']} g/day

Food Plan:
{result['FoodPlan']}
"""
        st.download_button("📄 Download Diet Report", diet_text, "diet_report.txt", "text/plain")

    else:
        st.warning("No match found for your selected criteria.")
        
elif page == "Physical Activity":
    activity_img = get_base64_image("static/activity.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{activity_img}");
            background-attachment: fixed;
            background-size: cover;
        }}

        h1, h2, h3, h4, h5, h6, p, label, .css-10trblm, .css-1d391kg, .css-1cpxqw2 {{
            color: black !important;
            font-size: 24px !important;
            font-weight: 600;
        }}

        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stTextArea>div>textarea {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: black !important;
            font-size: 18px !important;
        }}

        .stButton>button {{
            font-size: 18px !important;
            font-weight: bold;
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("🏃 Physical Activity Plan")

    risk = st.selectbox("🩺 Risk Level", ["🔴 High", "🟢 Low"])
    age = st.number_input("🎂 Age", 1, 120)
    weight = st.number_input("⚖ Weight (kg)", 30.0, 200.0)
    gender = st.selectbox("🧑 Gender", ["👨 Male", "👩 Female"])

    result = get_personalized_exercise(risk, age, gender, weight)

    # Age-based YouTube video mapping
    if age < 18:
        video_link = "https://www.youtube.com/watch?v=4pKly2JojMw"
    elif 18 <= age < 40:
        video_link = "https://www.youtube.com/watch?v=hJbRpHZr_d0"
    elif 40 <= age < 60:
        video_link = "https://www.youtube.com/watch?v=GskHIA3iE5Q"
    elif 60 <= age < 80:
        video_link = "https://www.youtube.com/watch?v=tiJocDzy2rc"
    else:
        video_link = "https://www.youtube.com/watch?v=1DYH5ud3zHo"

    st.subheader("🏋 Activity Level")
    st.write(result.get("ActivityLevel", "-"))

    st.subheader("🌞 Morning")
    st.write(result.get("Morning", "-"))

    st.subheader("🌤 Afternoon")
    st.write(result.get("Afternoon", "-"))

    st.subheader("🌆 Evening")
    st.write(result.get("Evening", "-"))

    st.subheader("📌 Note")
    st.info(result.get("Note", "-"))

    st.subheader("🎥 Recommended Exercise Video")
    st.video(video_link)

    # Save to report state
    st.session_state.report.update({
        "Activity Level": result.get("ActivityLevel", "-"),
        "Morning Exercise": result.get("Morning", "-"),
        "Afternoon Exercise": result.get("Afternoon", "-"),
        "Evening Exercise": result.get("Evening", "-")
    })

    activity_text = f"""PHYSICAL ACTIVITY REPORT
Activity Level: {result.get('ActivityLevel', '-')}
Morning: {result.get('Morning', '-')}
Afternoon: {result.get('Afternoon', '-')}
Evening: {result.get('Evening', '-')}
"""
    st.download_button("📄 Download Activity Report", activity_text, "activity_report.txt", "text/plain")

elif page == "Diagnostic Tests":
    diag_img = get_base64_image("static\diagnostic.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{diag_img}");
            background-attachment: fixed;
            background-size: cover;
        }}

        h1, h2, h3, h4, h5, h6, p, label, .css-10trblm, .css-1d391kg, .css-1cpxqw2 {{
            color: black !important;
            font-size: 24px !important;
            font-weight: 600;
        }}

        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stTextArea>div>textarea {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: black !important;
            font-size: 18px !important;
        }}

        .stButton>button {{
            font-size: 18px !important;
            font-weight: bold;
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # Diagnostic content begins
    st.header("🧪 Diagnostic Tests & Interpretation")

    risk = st.selectbox("What was your predicted Diabetes Risk?", ["High", "Low", "Already Diabetic"])
    if risk == "Low":
        st.success("Your risk is low. Routine diagnostic tests aren't necessary right now.")
        st.info("Keep maintaining a healthy lifestyle. No immediate concern.")
    else:
        st.info("Please enter your diagnostic test results below:")

        fbs = st.number_input("FBS - Fasting Blood Sugar (mg/dL)", 50, 300)
        ppbs = st.number_input("PPBS - Post Prandial Blood Sugar (mg/dL)", 50, 300)
        rbs = st.number_input("RBS - Random Blood Sugar (mg/dL)", 50, 300)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, step=0.1)

        def interpret_fbs(val):
            if val < 100: return "Normal"
            elif val < 126: return "Pre-Diabetes"
            else: return "Diabetes"

        def interpret_ppbs(val):
            if val < 140: return "Normal"
            elif val < 200: return "Pre-Diabetes"
            else: return "Diabetes"

        def interpret_rbs(val):
            return "Normal" if val < 140 else "Diabetes"

        def interpret_hba1c(val):
            if val < 5.7: return "Normal"
            elif val < 6.5: return "Pre-Diabetes"
            else: return "Diabetes"

        st.markdown("### 🧾 Test Interpretations")
        st.write(f"*FBS*: {interpret_fbs(fbs)}")
        st.write(f"*PPBS*: {interpret_ppbs(ppbs)}")
        st.write(f"*RBS*: {interpret_rbs(rbs)}")
        st.write(f"*HbA1c*: {interpret_hba1c(hba1c)}")

        results = [
            interpret_fbs(fbs),
            interpret_ppbs(ppbs),
            interpret_rbs(rbs),
            interpret_hba1c(hba1c)
        ]

        if "Diabetes" in results:
            msg = ("Your values suggest poor control. Please follow your treatment and "
                   "consult your doctor.") if risk == "Already Diabetic" else \
                  "⚠ Your results indicate *Diabetes*. Consult a healthcare provider immediately."
            st.error(msg)
        elif "Pre-Diabetes" in results:
            st.warning("You're in the *Pre-Diabetic* range. Improve your lifestyle and retest in 3 months.")
        else:
            st.success("✅ All test values are within *normal range*. Keep up your good habits!")


elif page == "Ask the Bot":
    bot_img = get_base64_image("static/chatbot.jpg")
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bot_img}");
        background-attachment: fixed;
        background-size: cover;
    }}

    h1, h2, h3, h4, h5, h6, p, li, label, .css-10trblm, .css-1d391kg, .css-1cpxqw2 {{
        color: white !important;
        font-size: 24px !important;
        font-weight: 600;
    }}

    ul, li {{
        color: black !important;
        font-size: 20px !important;
        font-weight: 500;
    }}

    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stTextArea>div>textarea {{
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: black !important;
        font-size: 18px !important;
    }}

    .stButton>button {{
        font-size: 18px !important;
        font-weight: bold;
        color: black !important;
    }}

    .stChatMessage .stMarkdown, .stChatMessage p, .stChatMessage span {{
        color: black !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    st.header("🤖 Ask the Diabetes Bot")
    question = st.text_input("Ask any diabetes-related question:")
    if question:
        with st.spinner("Thinking..."):
            cohere_answer = ask_cohere_bot(question)
            st.info("🤖 Answer from Cohere AI:")
            st.success(cohere_answer)
            
elif page == "Patient Report":
    patient_img = get_base64_image("static/final report.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{patient_img}");
            background-attachment: fixed;
            background-size: cover;
        }}
        h1, h2, h3, h4, h5, h6, p, label {{
            color: black !important;
            font-size: 24px !important;
            font-weight: 600;
        }}
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stTextArea>div>textarea {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: black !important;
            font-size: 18px !important;
        }}
        .stButton>button {{
            font-size: 18px !important;
            font-weight: bold;
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("📋 Patient Report")
    st.markdown("Upload your prediction, diet, and activity reports below to generate a full patient summary.")

    # Uploads
    pred_file = st.file_uploader("📄 Upload Prediction Report", type="txt")
    diet_file = st.file_uploader("📄 Upload Diet & Nutrition Report", type="txt")
    activity_file = st.file_uploader("📄 Upload Physical Activity Report", type="txt")

    if pred_file and diet_file and activity_file:
        pred_text = pred_file.read().decode("utf-8")
        diet_text = diet_file.read().decode("utf-8")
        activity_text = activity_file.read().decode("utf-8")

        full_report = f"""DIABETES PATIENT REPORT

--- PREDICTION ---
{pred_text.strip()}

--- DIET & NUTRITION ---
{diet_text.strip()}

--- PHYSICAL ACTIVITY ---
{activity_text.strip()}
"""
        st.success("✅ All files uploaded successfully.")
        st.text_area("📋 Combined Report Preview", value=full_report, height=400)

        st.download_button("📥 Download Full Combined Report", full_report, "full_patient_report.txt", "text/plain")
    else:
        st.warning("⚠ Please upload all three reports to generate the final combined report.")

elif page == "Emergency Guide":
    emergency_img = get_base64_image("static/emergency.jpg")  # Optional background
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{emergency_img}");
            background-attachment: fixed;
            background-size: cover;
        }}
        h2, h3, p, li {{
            color: black !important;
            font-size: 20px;
        }}
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<h2 style='font-size:36px;'>🆘 Emergency Guide for Diabetics</h2>", unsafe_allow_html=True)
    st.markdown("This quick-reference guide provides life-saving steps for diabetic emergencies.", unsafe_allow_html=True)

    st.markdown("### ⛔ Hypoglycemia (Low Blood Sugar)")
    st.markdown("""
- *Symptoms:* Shaking, sweating, dizziness, confusion, loss of consiousness.
- *Immediate Actions:*
    - Give 15g of fast-acting carbs: 1/2 glass juice, glucose tablets, sugar candy.
    - Recheck sugar after 15 mins.
    - If still low,Call emergency services.
    """)

    st.markdown("### 🔺 Hyperglycemia (High Blood Sugar)")
    st.markdown("""
- *Symptoms:* Frequent urination, thirst, fatigue, fruity breath, nausea.
- *Immediate Actions:*
    - Drink water to stay hydrated.
    - Avoid sugary or high-carb food.
    - If the above symptoms persist call emergency services immediately.
    """)

    st.markdown("### ☎ When to Seek Emergency Help")
    st.markdown("""
- Unconsciousness
- Seizure
- Severe vomiting or dehydration
- Confusion or disorientation
""")

    # Text for download
    emergency_text = """EMERGENCY GUIDE FOR DIABETICS

⛔ Hypoglycemia (Low Blood Sugar)
Symptoms: Shaking, sweating, dizziness, confusion.
Actions:
- Take 15g of sugar (juice, candy).
- Recheck in 15 mins.
- Repeat if needed.
- If unconscious: call emergency, do NOT feed.

🔺 Hyperglycemia (High Blood Sugar)
Symptoms: Thirst, frequent urination, fruity breath.
Actions:
- Drink water.
- Avoid carbs.
- Call doctor if sugar >300 mg/dL.

☎ Emergency Signs:
- Unconscious
- Seizure
- Confusion
- Vomiting

Keep this guide handy. Share with caregivers.
"""

    st.download_button("📄 Download Emergency Guide", emergency_text, "emergency_guide.txt", "text/plain")
