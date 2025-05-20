import streamlit as st
from tools import ResumeTool
from agent import build_agent
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="AI HR Assistant", layout="wide")
st.title("AI HR Assistant for Resume Screening")

resume_tool = ResumeTool()

uploaded_resumes = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
job_description = st.text_area("Paste the Job Description")

if uploaded_resumes:
    msg = resume_tool.load_resumes(uploaded_resumes)
    st.success(msg)

if job_description:
    st.success(resume_tool.set_job_description(job_description))

if uploaded_resumes and job_description:
    if "agent" not in st.session_state:
        st.session_state.agent = build_agent(resume_tool)

    user_input = st.text_input("Ask a question (e.g., Who are the top candidates?)")
    if user_input:
        response = st.session_state.agent.run(user_input)
        st.markdown(response)
