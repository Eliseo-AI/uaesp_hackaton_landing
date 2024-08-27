import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Configuración del prompt template
prompt_template = """

Instrucciones
Eres un analista de datos que hace informes, responde preguntas de un documento en español para una oficina gubernamental especializada en servicios pùblicos de Alumbrado público, agua potable y alcantarrilado,
, Luz, Recolección de bauras. Este capaz de responder en español a preguntas especificas de indicadores y puede dar una respuesta de contexto sobre un informe

Ejemplos de tareas que el analista de datos  debe poder realizar:

Ayuda con cantidades y medidas: El usuario puede preguntar sobre medidas o cantidades especificas en número, porcentajes, 
 y el asistente debe proporcionar información sobre su ese indicador o medida que se le pidio, puede citar la documentación que tiene.

Ejemplo de preguntas que no se debe responder:
Quien era el alcalde de bogotá en una fecha especifica?.
Porque los servicios públicos son caros?.

Ejemplo de Pregunta y respuesta:
En caso que el usuario escriba palabras ofensivas, soeces o groserías, se le recomienda de manera respetuosa dejar el chatbot  y 
tener un momento de meditación.

Explicación de conceptos: El usuario puede necesitar comprender conceptos específicos de alguno de los indicadores o conocer una palabra en 
especial o técnica para comprender una medición, si la información esta disponible en el documento proveerla. 

Las respuestas deben ser en español.
Si alguien pregunta en otro idioma, se le contesta en español.

Question:{question}

Documents: {context}

"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


loader = PyPDFDirectoryLoader("/home/criuser/Documents/2024/UAESP/Docs/")
docs = loader.load()

# Dividir los documentos en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)

# Configurar embeddings de Cohere
cohere_api_key = 'vuU6GAZ7fRgb1LGW9XXO25zvsBX42RImfPqpjOm5'
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0", 
    cohere_api_key=cohere_api_key
)

# Crear el vectorstore usando FAISS
vectorstore = FAISS.from_documents(texts, embeddings)

# Configurar la cadena de QA con Cohere
qa = RetrievalQA.from_chain_type(
    llm=Cohere(model="command-nightly", temperature=0, cohere_api_key=cohere_api_key, max_tokens=1000),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=False,
    chain_type_kwargs={
        "verbose": False,
        "prompt": PROMPT,
        "memory": ConversationBufferMemory(memory_key="history", input_key="question"),
    },
)

# Configuración de la interfaz de Streamlit
st.title("Chatbot de Documentos Gubernamentales")
st.write("Haz una pregunta sobre los servicios públicos y te responderé con base en la documentación disponible.")

# Campo de entrada de la pregunta del usuario
user_question = st.text_input("Escribe tu pregunta aquí:")

# Botón para enviar la pregunta
if st.button("Enviar"):
    if user_question:
        # Obtener la respuesta del modelo
        answer = qa.run({"query": user_question})
        st.write("**Respuesta:**", answer)
    else:
        st.write("Por favor, escribe una pregunta.")

