import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeTool:
    def __init__(self):
        self.resumes = {}
        self.job_description = ""

    def load_resumes(self, files):
        for file in files:
            text = ""
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            self.resumes[file.name] = text
        return f"Loaded {len(self.resumes)} resume(s)."

    def set_job_description(self, text):
        self.job_description = text
        return "Job description saved."

    def rank_candidates(self):
        if not self.resumes or not self.job_description:
            return "Please upload resumes and set a job description first."

        vectorizer = TfidfVectorizer(stop_words="english")
        documents = [self.job_description] + list(self.resumes.values())
        tfidf_matrix = vectorizer.fit_transform(documents)

        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        ranked = sorted(zip(self.resumes.keys(), scores), key=lambda x: x[1], reverse=True)

        result = "### 🏆 Ranked Candidates:\n"
        for i, (name, score) in enumerate(ranked, start=1):
            result += f"{i}. **{name}** — Relevance Score: {score:.2f}\n"
        return result
