## About Dataset

This dataset comprises comprehensive health information for patients who underwent examination to get diagnosed with Parkinson's Disease, each uniquely identified with IDs ranging from 3058 to 5162. The dataset includes demographic details, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, and symptoms.

## Table of Contents

1. **Patient Information**
   - Patient ID
   - Demographic Details
   - Lifestyle Factors
2. **Medical History**
3. **Clinical Measurements**
4. **Cognitive and Functional Assessments**
5. **Symptoms**
6. **Confidential Information**

## Patient Information

### Patient ID

- **PatientID**: A unique identifier assigned to each patient (3058 to 5162).

### Demographic Details

- **Age**: The age of the patients ranges from 50 to 90 years.
- **Gender**: Gender of the patients.
- **Ethnicity**: The ethnicity of the patients.  (تتعلق بالثقافة واللغة والتقاليد المشتركة)
- **EducationLevel**: The education level of the patients.

### Lifestyle Factors

- **BMI**: Body Mass Index of the patients. 
         🧭 BMI Categories (for adults):
         Below 18.5 → Underweight

         18.5 – 24.9 → Normal weight

         25 – 29.9 → Overweight

         30 and above → Obese
-----------------------------------------------------------------
- **Smoking**: Smoking status. ( yes / no)
- **AlcoholConsumption**: Weekly alcohol consumption in units. (+20 units is very high level)

- **PhysicalActivity**: Weekly physical activity in hours. 
(doing physical activity per week, measured in hours and minutes (formatted as HH:MM).)

- **DietQuality**: Diet quality score. (from 1 to 10 koll lma nzed kol lma ykon a7sn)
- **SleepQuality**: Sleep quality score. (from 1 to 10 koll lma nzed kol lma ykon a7sn)

## Medical History

- **FamilyHistoryParkinsons**: Family history of Parkinson's Disease.
- **TraumaticBrainInjury**: History of traumatic brain injury.
- **Hypertension**: Presence of hypertension.(High  blood preasusre)
- **Diabetes**: Presence of diabetes.(sugar)
- **Depression**: Presence of depression.(اكتئاب)
- **Stroke**: History of stroke.

## Clinical Measurements

- **SystolicBP**: Systolic blood pressure, ranging from 90 to 180 mmHg.

      Systolic BP (Blood Pressure):

      This is the pressure in the arteries when the heart beats.

      Healthy Range: 90–120 mmHg is considered normal. A systolic BP of 120–129 mmHg is considered elevated, and anything over 130 mmHg is high blood pressure (hypertension).

      Too High: A systolic BP above 140 mmHg is considered hypertensive crisis.
------------------------------------------------------------------------------------


- **DiastolicBP**: Diastolic blood pressure, ranging from 60 to 120 mmHg.

Diastolic BP (Blood Pressure):

This is the pressure in the arteries when the heart is at rest between beats.

Healthy Range: 60–80 mmHg is considered normal. Anything above 80 mmHg is elevated or hypertensive.

Too High: A diastolic BP over 90 mmHg is hypertensive.

------------------------------------------------------------------------------------------------
- **CholesterolTotal**: Total cholesterol levels, ranging from 150 to 300 mg/dL.

This is the overall amount of cholesterol in the blood, including LDL, HDL, and triglycerides.

Healthy Range: Total cholesterol should be below 200 mg/dL. Levels between 200–239 mg/dL are borderline high, and above 240 mg/dL is high.

Too High: Over 240 mg/dL can increase cardiovascular disease risk.
--------------------------------------------------------------------------------------------------

- **CholesterolLDL**: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL.

Often called "bad" cholesterol, it can build up in arteries and increase the risk of heart disease.

Healthy Range: Less than 100 mg/dL is optimal. 100–129 mg/dL is near optimal, while 130–159 mg/dL is borderline high, and anything over 160 mg/dL is high.

Too High: A value over 160 mg/dL is considered high and can significantly increase heart disease risk.

-----------------------------------------------------------------------------------------------------------

- **CholesterolHDL**: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL.

Known as "good" cholesterol, HDL helps remove excess cholesterol from the blood.

Healthy Range: 40–60 mg/dL is considered good. Levels over 60 mg/dL are ideal and protective against heart disease.

Too Low: Less than 40 mg/dL is considered a risk factor for cardiovascular disease.

---------------------------------------------------------------------------------------------

- **CholesterolTriglycerides**: Triglycerides levels, ranging from 50 to 400 mg/dL.

Triglycerides are a type of fat found in the blood. High levels are linked to cardiovascular disease.

Healthy Range: Less than 150 mg/dL is normal. 150–199 mg/dL is borderline high, 200–499 mg/dL is high, and 500 mg/dL or more is very high.

Too High: Over 200 mg/dL is considered high, which increases the risk of heart disease and other health problems.

------------------------------------------------------------------------------------------------

## Cognitive and Functional Assessments

- **UPDRS (target variable)** : Unified Parkinson's Disease Rating Scale score, ranging from 0 to 199. Higher scores indicate greater severity of the disease.


- **MoCA**: Montreal Cognitive Assessment score, ranging from 0 to 30. Lower scores indicate cognitive impairment. (koll lma yzed kol lma ykon a7sn)

MoCA (Montreal Cognitive Assessment) هو اختبار يستخدم لتقييم الوظائف  (الذهنية) لدى الأفراد، ويُستخدم بشكل شائع للكشف عن الاضطرابات المعرفية مثل الخرف أو فقدان الذاكرة. يتم استخدامه أيضًا في متابعة مرضى مرض باركنسون لتقييم تأثير المرض على القدرات العقلية.

نطاق درجات MoCA:
الدرجة تتراوح من 0 إلى 30.

30: الدرجة المثالية تعني أن الشخص لا يعاني من أي مشاكل معرفية.

أقل من 26: يشير إلى احتمال وجود تدهور معرفي أو مشكلات في الذاكرة أو التفكير.

---------------------------------------------------------------------------------------------------

- **FunctionalAssessment**: Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.

الدرجة تتراوح من 0 إلى 10.

10: الدرجة الأعلى تعني أن الشخص قادر على أداء الأنشطة اليومية بشكل جيد دون أي مشاكل.

0: الدرجة الأدنى تشير إلى وجود إعاقة شديدة تؤثر بشكل كبير على قدرة الشخص في أداء الأنشطة اليومية.
------------------------------------------------------------------------------------------------------

## Symptoms

- **Tremor**: Presence of tremor. (ارتعاش)
- **Rigidity**: Presence of muscle rigidity.
(التصلب هو حالة تكون فيها العضلات قاسية أو مشدودة بشكل غير طبيعي، حتى عندما تكون في حالة راحة.

في مرض باركنسون، يحدث التصلب بسبب تأثير المرض على الجهاز العصبي، مما يجعل العضلات غير قادرة على الاسترخاء بشكل طبيعي.)

- **Bradykinesia**: Presence of bradykinesia (slowness of movement).
- **PosturalInstability**: Presence of postural instability.
(هو فقدان التوازن أو الصعوبة في الحفاظ على وضعية الجسم المستقيمة.)

- **SpeechProblems**: Presence of speech problems.(مشاكل فى الكلام)

- **SleepDisorders**: Presence of sleep disorders.(أظطرابات فى النوم)

- **Constipation**: Presence of constipation.(إمساك)

## Confidential Information

- **DoctorInCharge**: This column contains confidential information about the doctor in charge, with "DrXXXConfid" as the value for all patients.

## Conclusion

This dataset offers extensive insights into the factors associated with Parkinson's Disease, including demographic, lifestyle, medical, cognitive, and functional variables. It is ideal for developing predictive models, conducting statistical analyses, and exploring the complex interplay of factors contributing to Parkinson's Disease.
