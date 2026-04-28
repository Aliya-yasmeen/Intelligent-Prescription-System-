from django.shortcuts import render

# Create your views here.
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from django.conf import settings
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load and preprocess data


def training(request):
    df = pd.read_csv(os.path.join(BASE_DIR, 'media', 'Drug_Data.csv'))
    df.dropna(subset=['Prescribed_for', 'Drug_Review', 'drugName'], inplace=True)
    df['input_text'] = df['Prescribed_for'] + " " + df['Drug_Review']
    # top_n = 100  # or 20, 10, etc.
    # top_classes = df['drugName'].value_counts().nlargest(top_n).index
    # df = df[df['drugName'].isin(top_classes)]

    label_encoder = LabelEncoder()
    df['drug_label'] = label_encoder.fit_transform(df['drugName'])
    joblib.dump(label_encoder, os.path.join(BASE_DIR, 'media', 'drug_label_encoder.pkl'))

    df_sample = df.sample(frac=0.2, random_state=42)
    X = df_sample['input_text']
    y = df_sample['drug_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, os.path.join(BASE_DIR, 'media', 'drug_model_pipeline.pkl'))

    y_pred = pipeline.predict(X_test)
    target_names = label_encoder.inverse_transform(sorted(set(y_test) | set(y_pred)))
    report = classification_report(y_test, y_pred, target_names=target_names, labels=sorted(set(y_test) | set(y_pred)))

    accuracy = accuracy_score(y_test, y_pred)
   


    return render(request, 'users/accuracy.html', {
        'accuracy': accuracy,
        'report': report
    })
    

from django.shortcuts import render
import pandas as pd
import joblib
import os
from django.conf import settings


def prediction(request):
    BASE_DIR = settings.BASE_DIR

    if request.method == 'POST':
        # Load model pipeline and label encoder
        pipeline = joblib.load(os.path.join(BASE_DIR, 'media', 'drug_model_pipeline.pkl'))
        label_encoder = joblib.load(os.path.join(BASE_DIR, 'media', 'drug_label_encoder.pkl'))

        # Get input values from form
        prescribed_for = request.POST.get('prescribed_for', '')
        drug_review = request.POST.get('drug_review', '')

        # Combine into single input text
        input_text = prescribed_for + " " + drug_review

        # Create input DataFrame
        input_df = pd.DataFrame({'input_text': [input_text]})

        # Predict
        prediction_encoded = pipeline.predict(input_df['input_text'])[0]
        predicted_drug = label_encoder.inverse_transform([prediction_encoded])[0]

        context = {
            'prescribed_for': prescribed_for,
            'drug_review': drug_review,
            'predicted_drug': predicted_drug
        }

        return render(request, 'users/prediction.html', context)

    return render(request, 'users/prediction.html')





# Create your views here.
import os

def ViewDataset(request):
    dataset = os.path.join(settings.MEDIA_ROOT, 'Drug_Data.csv')
    import pandas as pd
    df = pd.read_csv(dataset, nrows=100)
    df = df.to_html(index=None)
    return render(request, 'users/viewData.html', {'data': df})


from django.shortcuts import render, redirect
from .models import UserRegistrationModel
from django.contrib import messages

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request,"Registration successful!")
    return render(request, 'UserRegistrations.html') 


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def index(request):
    return render(request,"index.html")
