import os
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify

# 🔹 Load OpenAI API Key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is missing! Check your .env file.")

# 🔹 Set up Tesseract OCR (for scanned PDFs)
# Note: This path is Windows-specific; we'll adjust for deployment
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔹 Function to Extract Text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts Hindi text from a PDF file (supports both text-based and scanned PDFs)."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    if not text.strip():
        print(f"🔍 No direct text found in {pdf_path}, using OCR...")
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img, lang="hin") + "\n"
    return text.strip()

# 🔹 Function to Extract Text from DOCX Files
def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text.strip()

# 🔹 Folder Path for PDFs & DOCX Files
data_folder = os.path.join(os.path.dirname(__file__), "data_files", "documents")
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Folder not found: {data_folder}")

# 🔹 Read Multiple PDFs & DOCX Files
all_text = []
for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)
    if filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
        if extracted_text:
            all_text.append(extracted_text)
            print(f"✅ Loaded PDF: {filename}")
    elif filename.endswith(".docx"):
        extracted_text = extract_text_from_docx(file_path)
        if extracted_text:
            all_text.append(extracted_text)
            print(f"✅ Loaded DOCX: {filename}")

if not all_text:
    raise ValueError("No valid content found in the PDFs or DOCX files.")
full_text = "\n".join(all_text)
print("✅ All PDFs & DOCX files loaded successfully!")

# 🔹 Split Text into Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(full_text)
print(f"✅ Generated {len(chunks)} text chunks.")

# 🔹 Generate Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# 🔹 Create FAISS Vector Store
vector_db = FAISS.from_texts(chunks, embedding_model)
print("✅ FAISS Vector Store Created!")

# 🔹 Build AI Agent with Retrieval
llm = OpenAI(api_key=openai_api_key)
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🔹 Handle User Input
def handle_user_input(user_query):
    """Retrieves answer from Hindi PDFs & DOCX and translates it into English."""
    hindi_response = qa_chain.run(user_query)
    if not hindi_response:
        return "I'm sorry, I couldn't find an answer to that question."
    translation_prompt = f"Translate the following Hindi text to English:\n\n{hindi_response}\n\nProvide a natural English answer."
    english_response = llm.invoke(translation_prompt)
    return english_response if english_response else "Translation failed."

# 🔹 Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_name = request.form.get("name")
        user_dob = request.form.get("dob")
        query = request.form.get("question")
        if user_name and user_dob and query:
            response = handle_user_input(query)
            greeting = f"🤖 Welcome, {user_name}! 🎉 I am here to assist you with queries from my trained knowledge."
            return render_template("index.html", greeting=greeting, response=response, name=user_name, dob=user_dob)
    return render_template("index.html", greeting="🤖 Hello! I am your AI assistant. Please provide your details to begin.")

@app.route("/exit")
def exit_app():
    name = request.args.get("name", "friend")
    farewell = f"👋 Bye, {name}! Have a great day! 😊"
    return jsonify({"farewell": farewell})

if _name_ == "__main__":
    app.run(debug=True)