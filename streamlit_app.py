
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

st.set_page_config(
    page_title="BNCC Computação | RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 BNCC Computação")
st.subheader("Assistente de perguntas e respostas sobre documentos")

st.write(
    "Faça uma pergunta sobre o documento da BNCC Computação. "
    "As respostas são baseadas nos trechos recuperados do documento."
)

with st.sidebar:
    st.header("⚙️ Configuração")
    st.write("Assistente de pesquisa com RAG")
    st.caption("Gemini + Pinecone")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("Configure GEMINI_API_KEY nos Secrets do Streamlit.")
    st.stop()

if "PINECONE_API_KEY" not in st.secrets:
    st.error("Configure PINECONE_API_KEY nos Secrets do Streamlit.")
    st.stop()

if "PINECONE_INDEX_NAME" not in st.secrets:
    st.error("Configure PINECONE_INDEX_NAME nos Secrets do Streamlit.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])

pergunta = st.text_area(
    "Digite sua pergunta:",
    placeholder="Quais são as habilidades do primeiro ano?",
    height=100
)

if st.button("Buscar resposta", type="primary"):
    if not pergunta.strip():
        st.warning("Digite uma pergunta antes de continuar.")
    else:
        with st.spinner("Pesquisando o documento..."):
            resultado = genai.embed_content(
                model="models/gemini-embedding-001",
                content=pergunta,
                task_type="retrieval_query"
            )

            query_embedding = resultado["embedding"]

            busca = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )

            matches = busca.get("matches", [])

            if not matches:
                st.warning("Não encontrei informações relevantes no documento.")
            else:
                trechos = [
                    match["metadata"]["texto_original"]
                    for match in matches
                    if match.get("metadata", {}).get("texto_original")
                ]

                contexto = "\n\n".join(trechos)

                if not contexto:
                    st.warning("Os resultados não contêm texto para responder.")
                else:
                    prompt = f"""
Responda em português usando somente as informações do contexto.
Se a resposta não estiver no contexto, diga:
"Informação indisponível no contexto."

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

                    modelo = genai.GenerativeModel("gemini-3.6-flash")
                    resposta = modelo.generate_content(prompt)

                    st.markdown("### 💡 Resposta")
                    st.write(resposta.text)

                    with st.expander("📄 Consultar trechos recuperados"):
                        for i, trecho in enumerate(trechos, start=1):
                            st.markdown(f"**Trecho {i}**")
                            st.write(trecho)
