import os
import shutil
import uuid
from datetime import datetime
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Configuración inicial
app = FastAPI(
    title="WhatsApp X Document Processor",
    description="API para procesar documentos de negocio y generar embeddings para el chatbot",
    version="1.0.0"
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios
UPLOAD_DIR = "uploaded_docs"
CHROMA_PERSIST_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Configuración de embeddings
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "business_docs"

# Configuración del splitter de texto
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

def get_loader(file_path: str, file_type: str):
    """Retorna el loader apropiado según el tipo de archivo"""
    file_type = file_type.lower()
    if file_type == "pdf":
        return PyPDFLoader(file_path)
    elif file_type == "docx":
        return Docx2txtLoader(file_path)
    elif file_type == "txt":
        return TextLoader(file_path)
    elif file_type in ["xlsx", "xls"]:
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

@app.post("/upload-docs/", response_model=Dict)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint para subir documentos de negocio (PDF, DOCX, XLSX, TXT)
    
    Args:
        files: Lista de archivos a subir
        
    Returns:
        Dict con información de los archivos subidos
    """
    try:
        file_paths = []
        for file in files:
            # Validar tipo de archivo
            file_ext = os.path.splitext(file.filename)[1][1:].lower()
            if file_ext not in ["pdf", "docx", "xlsx", "xls", "txt"]:
                continue
            
            # Generar nombre único para el archivo
            unique_filename = f"{uuid.uuid4()}.{file_ext}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Guardar el archivo
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append({
                "original_name": file.filename,
                "saved_path": file_path,
                "file_type": file_ext.upper(),
                "upload_date": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path)
            })
        
        if not file_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid documents were uploaded (supported: PDF, DOCX, XLSX, TXT)"
            )
        
        return JSONResponse({
            "status": "success",
            "message": "Documents uploaded successfully",
            "files": file_paths,
            "next_step": {
                "process_endpoint": "/process-docs/",
                "method": "POST"
            }
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading documents: {str(e)}"
        )

@app.post("/process-docs/", response_model=Dict)
async def process_documents():
    """
    Endpoint para procesar documentos subidos y generar embeddings
    
    Returns:
        Dict con resultados del procesamiento
    """
    try:
        # 1. Cargar documentos
        documents = []
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            file_ext = os.path.splitext(filename)[1][1:].lower()
            
            try:
                loader = get_loader(file_path, file_ext)
                loaded_docs = loader.load()
                
                # Dividir documentos en chunks
                split_docs = TEXT_SPLITTER.split_documents(loaded_docs)
                documents.extend(split_docs)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No valid documents found for processing"
            )
        
        # 2. Crear embeddings y almacenar en Chroma
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )
        vectordb.persist()
        
        # Estadísticas
        total_chunks = len(documents)
        unique_sources = len(set(doc.metadata.get("source", "") for doc in documents))
        
        return JSONResponse({
            "status": "success",
            "message": "Documents processed and stored in ChromaDB",
            "stats": {
                "total_chunks": total_chunks,
                "unique_documents": unique_sources,
                "collection_name": CHROMA_COLLECTION_NAME,
                "embedding_model": EMBEDDINGS_MODEL
            },
            "chroma_info": {
                "persist_directory": CHROMA_PERSIST_DIR,
                "api_endpoint": "http://localhost:8001/api/v1"
            }
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )

@app.get("/health/", response_model=Dict)
async def health_check():
    """Endpoint para verificar el estado del servicio"""
    return {
        "status": "healthy",
        "services": {
            "document_upload": "active",
            "chroma_db": "connected" if os.path.exists(CHROMA_PERSIST_DIR) else "inactive"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)