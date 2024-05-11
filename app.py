import streamlit as st
import re
import pickle

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(txt):
    cleaned_text = re.sub(r'http\S+', ' ', txt)
    cleaned_text = re.sub('_', ' ', cleaned_text)
    cleaned_text = re.sub(r'RT|CC', ' ', cleaned_text)
    cleaned_text = re.sub(r'#\S+\s', ' ', cleaned_text)
    cleaned_text = re.sub(r'@\S+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    cleaned_text = re.sub(r'[^\x00-\x7f]', ' ', cleaned_text)
    return cleaned_text

# create website
def main():
    st.title("Resume Analyser")
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'txt'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        predictions = clf.predict(input_features)
        st.write(predictions)

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(predictions[0], "Unknown")

        # for print  the result in web page
        st.write("Predicted Category :", category_name)


if __name__ == "__main__":
    main()
