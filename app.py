import os
import io
from gtts import gTTS
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import pandas as pd  # For saving responses in .xlsx format
from langchain.docstore.document import Document
import base64
from datetime import datetime
from langchain.prompts import PromptTemplate
import json
from typing import Dict, List

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def set_custom_style():
    """Set custom CSS styling for the Streamlit app."""
    st.markdown("""
        <style>
        /* CSS Variables */
        :root {
            --accent-color: #1e90ff;
            --background-color: #ffffff;
            --surface-color: #f8fafc;
            --border-color: #e2e8f0;
            --text-color-dark: #1e293b;
            --text-color-light: #64748b;
        }
        
        /* Main background */
        .stApp {
            background: var(--background-color) !important;
        }
        
        /* Chat messages */
        .stChatMessage {
            background-color: var(--surface-color) !important;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px;
            margin: 8px 0;
        }
        
        /* Buttons */
        .stButton button {
            background-color: var(--accent-color) !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            transition: opacity 0.3s !important;
        }
        .stButton button:hover {
            opacity: 0.9;
        }
        
        /* File uploader */
        .stUploadButton {
            background-color: var(--accent-color) !important;
            color: white !important;
            border-radius: 6px !important;
        }
        
        /* Text input */
        .stTextInput input {
            background-color: var(--background-color) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-color-dark) !important;
            border-radius: 6px !important;
        }
        
        /* Select box */
        .stSelectbox select {
            background-color: var(--background-color) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-color-dark) !important;
            border-radius: 6px !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--text-color-dark) !important;
        }
        
        /* Text color */
        .stMarkdown {
            color: var(--text-color-light);
        }
        
        /* Success message */
        .stSuccess {
            background-color: #f0fdf4 !important;
            color: #166534 !important;
            border: 1px solid #bbf7d0 !important;
        }
        
        /* Divider */
        hr {
            border-color: var(--border-color) !important;
        }
        
        /* Chat input container */
        .stChatInputContainer {
            background-color: var(--surface-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--surface-color);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--accent-color);
            border-radius: 4px;
            opacity: 0.5;
        }
        ::-webkit-scrollbar-thumb:hover {
            opacity: 0.7;
        }

        /* Chat message avatars */
        .stChatMessage.user .stChatMessageAvatar {
            background-color: var(--accent-color) !important;
        }
        .stChatMessage.assistant .stChatMessageAvatar {
            background-color: var(--text-color-light) !important;
        }
        </style>
    """, unsafe_allow_html=True)

def load_document(file_path):
    """Load a PDF document using PyMuPDF and extract text content.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of Document objects containing extracted text.
    """
    with fitz.open(file_path) as doc:
        text = "".join([page.get_text("text") for page in doc])
    return [Document(page_content=text)]

def process_uploaded_files(uploaded_files):
    """Process multiple uploaded PDF files.

    Args:
        uploaded_files (list): List of uploaded PDF files.

    Returns:
        list: Combined list of Document objects from all PDFs.
    """
    all_documents = []
    for uploaded_file in uploaded_files:
        file_path = f"{working_dir}/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        documents = load_document(file_path)
        all_documents.extend(documents)
    return all_documents

def setup_vectorstore(documents):
    """Create a FAISS vector store from the given documents.

    Args:
        documents (list): List of Document objects.

    Returns:
        FAISS: Initialized FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(doc_chunks, embeddings)

def get_response(vectorstore, question, chat_history):
    """Generate a chatbot response using vector store and ChatGroq.

    Args:
        vectorstore (FAISS): FAISS vector store for document search.
        question (str): User query.
        chat_history (list): List of chat history dictionaries.

    Returns:
        tuple: Assistant response and TTS audio stream.
    """
    docs = vectorstore.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])

    full_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    full_prompt += f"\nUser: {question}\nContext: {context}\nAssistant:"

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke(full_prompt)
    assistant_response = response.content

    tts = gTTS(assistant_response, lang="en")
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)

    return assistant_response, audio_fp

def save_chat_history(chat_history, file_type="txt"):
    """Save chat history to a file.

    Args:
        chat_history (list): List of chat messages.
        file_type (str): File type to save ('txt', 'pdf', or 'xlsx').
    """
    if file_type == "txt":
        with open("chat_history.txt", "w") as f:
            for msg in chat_history:
                f.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
    elif file_type == "xlsx":
        df = pd.DataFrame(chat_history)
        df.to_excel("chat_history.xlsx", index=False)
    elif file_type == "pdf":
        from reportlab.pdfgen import canvas
        c = canvas.Canvas("chat_history.pdf")
        y = 800
        for msg in chat_history:
            c.drawString(50, y, f"{msg['role'].capitalize()}: {msg['content']}")
            y -= 20
        c.save()

# Initialize templates
TEMPLATES = {
    "FIR": """
    First Information Report (FIR)
    Date: {date}
    FIR No.: {fir_number}
    Police Station: {police_station}
    
    Complainant Details:
    Name: {complainant_name}
    Address: {complainant_address}
    Contact: {complainant_contact}
    
    Incident Details:
    Date & Time of Incident: {incident_datetime}
    Place of Incident: {incident_place}
    Nature of Offense: {offense_nature}
    
    Brief Description:
    {incident_description}
    
    Accused Details (if known):
    Name: {accused_name}
    Description: {accused_description}
    
    Investigating Officer:
    Name: {officer_name}
    Rank: {officer_rank}
    Badge No.: {badge_number}
    """,
    
    "Investigation_Diary": """
    Investigation Diary
    Case No.: {case_number}
    Date: {date}
    
    Investigation Details:
    Time Started: {investigation_start_time}
    Location Visited: {location_visited}
    
    Actions Taken:
    {actions_taken}
    
    Evidence Collected:
    {evidence_collected}
    
    Witnesses Interviewed:
    {witnesses_interviewed}
    
    Next Steps:
    {next_steps}
    
    Investigating Officer:
    Name: {officer_name}
    Rank: {officer_rank}
    Badge No.: {badge_number}
    """,
    
    "Charge_Sheet": """
    Charge Sheet
    Case No.: {case_number}
    Date: {date}
    Police Station: {police_station}
    
    Accused Details:
    Name: {accused_name}
    Address: {accused_address}
    Age: {accused_age}
    
    Charges:
    1. {charges}
    
    Evidence Summary:
    {evidence_summary}
    
    Witness List:
    {witness_list}
    
    Investigation Officer:
    Name: {officer_name}
    Rank: {officer_rank}
    Badge No.: {badge_number}
    """,
    
    "Case_Closure": """
    Case Closure Report
    Case No.: {case_number}
    Date: {date}
    Police Station: {police_station}
    
    Case Status: {case_status}
    
    Summary of Investigation:
    {investigation_summary}
    
    Reason for Closure:
    {closure_reason}
    
    Final Disposition:
    {final_disposition}
    
    Investigating Officer:
    Name: {officer_name}
    Rank: {officer_rank}
    Badge No.: {badge_number}
    """
}

def setup_vectorstore():
    """Initialize the vector store with templates"""
    embeddings = HuggingFaceEmbeddings()
    documents = [
        Document(page_content=template, metadata={"type": doc_type})
        for doc_type, template in TEMPLATES.items()
    ]
    return FAISS.from_documents(documents, embeddings)

def generate_report(report_type, case_details, vectorstore):
    """Generate a report using RAG"""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    
    # Get relevant template
    docs = vectorstore.similarity_search(f"template for {report_type}", k=1)
    template = docs[0].page_content
    
    # Create prompt
    prompt = PromptTemplate(
        template="""You are a professional report writer for the police department.
        Using the following template and case details, generate a formal police report.
        Make sure to maintain proper formatting and professional language.
        
        Template:
        {template}
        
        Case Details:
        {case_details}
        
        Generated Report:""",
        input_variables=["template", "case_details"]
    )
    
    # Generate report
    response = llm.invoke(prompt.format(
        template=template,
        case_details=json.dumps(case_details, indent=2)
    ))
    
    return response.content

def extract_case_details(uploaded_file, report_type: str) -> Dict:
    """Extract relevant case details from uploaded document using LLM."""
    try:
        # Read PDF content with error handling
        pdf_content = ""
        pdf_bytes = uploaded_file.read()
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File size: {len(pdf_bytes)} bytes")
        
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    pdf_content += text
                    st.write(f"Page {page_num + 1} extracted text length: {len(text)}")
            
            if not pdf_content.strip():
                st.warning("No text content found in the PDF. The document might be scanned or image-based.")
                return {}
                
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return {}

        # Define the expected structure based on report type
        field_structures = {
            "FIR": {
                "case_number": "",
                "police_station": "",
                "complainant_name": "",
                "complainant_address": "",
                "complainant_contact": "",
                "incident_datetime": "",
                "incident_place": "",
                "offense_nature": "",
                "incident_description": "",
                "accused_name": "",
                "accused_description": "",
                "officer_name": "",
                "officer_rank": "",
                "badge_number": ""
            },
            "Investigation_Diary": {
                "case_number": "",
                "investigation_start_time": "",
                "location_visited": "",
                "actions_taken": "",
                "evidence_collected": "",
                "witnesses_interviewed": "",
                "next_steps": "",
                "officer_name": "",
                "officer_rank": "",
                "badge_number": ""
            },
            "Charge_Sheet": {
                "case_number": "",
                "accused_name": "",
                "accused_address": "",
                "accused_age": "",
                "charges": "",
                "evidence_summary": "",
                "witness_list": "",
                "officer_name": "",
                "officer_rank": "",
                "badge_number": ""
            },
            "Case_Closure": {
                "case_number": "",
                "case_status": "",
                "investigation_summary": "",
                "closure_reason": "",
                "final_disposition": "",
                "officer_name": "",
                "officer_rank": "",
                "badge_number": ""
            }
        }

        # Create the prompt with explicit instructions
        prompt = f"""Analyze the following police document and extract information into a structured format.
        Return ONLY a JSON object with the following fields (use empty strings for missing information):
        
        {json.dumps(field_structures[report_type], indent=2)}

        Document content:
        {pdf_content}

        Remember:
        1. Return ONLY the JSON object, no additional text
        2. Use empty strings for missing information
        3. Maintain the exact field names as shown above
        4. Ensure valid JSON format
        """

        # Call LLM with system prompt
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        
        response = llm.invoke(prompt)
        
        # Debug LLM response
        st.write(f"LLM Response: {response.content[:500]}...")  # Log the response for debugging
        
        try:
            # Attempt to extract JSON object from response
            json_str = response.content.strip()
            json_start = json_str.find('{')
            json_end = json_str.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = json_str[json_start:json_end]
                extracted_data = json.loads(json_str)
            else:
                st.error("Invalid JSON format received from LLM.")
                return field_structures[report_type]  # Return empty structure
            
            # Validate the structure
            expected_fields = set(field_structures[report_type].keys())
            actual_fields = set(extracted_data.keys())
            
            if expected_fields != actual_fields:
                missing = expected_fields - actual_fields
                extra = actual_fields - expected_fields
                if missing:
                    st.warning(f"Missing fields in extraction: {', '.join(missing)}")
                if extra:
                    st.warning(f"Extra fields in extraction: {', '.join(extra)}")
                
                # Ensure the correct structure
                validated_data = field_structures[report_type].copy()
                for field in expected_fields:
                    if field in extracted_data:
                        validated_data[field] = extracted_data[field]
                
                return validated_data
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {str(e)}")
            st.error("Raw response: " + response.content[:200])
            return field_structures[report_type]  # Return empty structure
            
    except Exception as e:
        st.error(f"Error in extraction process: {str(e)}")
        return field_structures[report_type]  # Return empty structure

def main():
    # Initialize vectorstore in session state if it doesn't exist
    if 'vectorstore' not in st.session_state:
        # Initialize with documents and embeddings
        documents = [
            Document(page_content=template, metadata={"type": doc_type})
            for doc_type, template in TEMPLATES.items()
        ]
        embeddings = HuggingFaceEmbeddings()
        st.session_state.vectorstore = FAISS.from_documents(documents, embeddings)

    # Set the page title and icon
    st.set_page_config(
        page_title="Police Report Auto-Fill System",
        page_icon="🚔",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Set the main title
    st.title("🚔 Police Report Auto-Fill System")

    # Introductory text
    st.markdown("""
        <p style='text-align: center; color: var(--text-color-light); margin-bottom: 2rem;'>
            Upload your PDF files and start generating police reports automatically.
        </p>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Report type selection
    st.markdown("### Select Report Type")
    report_type = st.selectbox(
        "Choose the type of report to generate",
        ["FIR", "Investigation_Diary", "Charge_Sheet", "Case_Closure"],
        format_func=lambda x: {
            "FIR": "First Information Report (FIR)",
            "Investigation_Diary": "Investigation Diary",
            "Charge_Sheet": "Charge Sheet",
            "Case_Closure": "Case Closure Report"
        }[x]
    )

    # File upload
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload case-related documents (PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload relevant case documents to extract information automatically"
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            # Process each document
            all_extracted_details = []
            
            for file in uploaded_files:
                st.info(f"Processing {file.name}...")
                extracted_details = extract_case_details(file, report_type)
                
                if extracted_details:
                    all_extracted_details.append(extracted_details)
                    st.success(f"Successfully extracted information from {file.name}")
            
            if all_extracted_details:
                # Show extracted information
                st.markdown("### Extracted Information")
                st.json(all_extracted_details[0])
                
                # Generate report
                if st.button("Generate Report"):
                    report = generate_report(report_type, all_extracted_details[0], st.session_state.vectorstore)
                    st.markdown("### Generated Report")
                    st.text_area("Report Content", report, height=400)
                    
                    # Download button
                    st.download_button(
                        "Download Report",
                        report,
                        file_name=f"{report_type}{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
                    )
            else:
                st.error("Could not extract information from any of the uploaded documents.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            assistant_response, audio_fp = get_response(st.session_state.vectorstore, user_input, st.session_state.chat_history)
            st.markdown(assistant_response)
            st.audio(audio_fp, format="audio/mp3")
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    if st.session_state.chat_history:  # Only show if there are messages
        st.markdown("<hr style='margin: 2rem 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            file_type = st.selectbox("Select file type", ["txt", "pdf", "xlsx"])
        with col2:
            if st.button("Save Chat History", use_container_width=True):
                save_chat_history(st.session_state.chat_history, file_type)
                st.success(f"Chat history saved as 'chat_history.{file_type}'!")

if __name__ == "__main__":
    main()