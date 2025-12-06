import streamlit as st
import json
import faiss
import numpy as np
from openai import OpenAI
from neo4j import GraphDatabase
from typing import List, Dict, Tuple
import os

class IndonesianLegalChatbot:
    def __init__(self):
        """Initialize the legal chatbot with all necessary connections and data"""
        # Initialize OpenAI client
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Load FAISS index and metadata for SEMANTIC search (content-based)
        self.faiss_index_path = st.secrets["paths"]["faiss_index"]
        self.metadata_path = st.secrets["paths"]["metadata"]
        
        self.index = faiss.read_index(self.faiss_index_path)
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load FAISS index and metadata for DIRECT search (name-based)
        self.faiss_nama_index_path = st.secrets["paths"]["faiss_nama_index"]
        self.metadata_nama_path = st.secrets["paths"]["metadata_nama"]
        
        self.nama_index = faiss.read_index(self.faiss_nama_index_path)
        with open(self.metadata_nama_path, 'r', encoding='utf-8') as f:
            self.metadata_nama = json.load(f)
        
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            st.secrets["neo4j"]["uri"],
            auth=(st.secrets["neo4j"]["user"], st.secrets["neo4j"]["password"])
        )
        
        # Configuration
        self.top_k_semantic = 3
        self.top_k_direct = 2
        self.similarity_threshold = 0.0001
        self.classifier_model = "gpt-4o-mini"
        self.reformer_model = "gpt-4o-mini"
        self.humanizer_model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"
    
    def classify_query(self, user_query: str) -> str:
        """Classify if query is 'direct' or 'semantic'"""
        classification_prompt = f"""You are a query classifier for an Indonesian legal system.

Classify the following query as either 'direct' or 'semantic':

DIRECT queries ask for specific articles/chapters by number:
- "apa bunyi pasal 2 uu no 2 tahun 2004"
- "pasal 15 perppu no 5 tahun 2020"
- "bab 3 uu no 123 tahun 2004"
- "pasal berapa yang mengatur tentang X di uu Y"

SEMANTIC queries ask conceptual questions:
- "apakah boleh merokok di tempat umum"
- "buatkan peraturan mengenai larangan merokok"
- "bagaimana hukuman untuk korupsi"
- "apa saja kewajiban perusahaan"

Query: {user_query}

Respond with ONLY one word: 'direct' or 'semantic'"""

        response = self.client.chat.completions.create(
            model=self.classifier_model,
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    
    def reform_direct_query(self, user_query: str) -> str:
        """Reform direct query to standardized format for name-based search"""
        reform_prompt = f"""You are an expert at reforming direct legal queries into standardized format.

Convert the user's query into this exact format: [DOCUMENT TYPE] Nomor [NUMBER] TAHUN [YEAR] PASAL [ARTICLE]

Document type abbreviations (MUST USE THESE EXACTLY):
- "Undang Undang" or "UU" → "UU"
- "Peraturan Pemerintah" or "PP" → "PP"
- "Peraturan Presiden" or "Perpres" → "PERPRES"
- "Peraturan Pemerintah Pengganti Undang Undang" or "Perppu" → "PERPPU"
- "Peraturan Menteri BUMN" or "Permen BUMN" → "PERMEN BUMN"
- "Peraturan Menteri" → "PERMEN"

Format rules:
1. Document type in UPPERCASE abbreviation
2. "Nomor" (not "no" or "No.")
3. "TAHUN" in UPPERCASE
4. "PASAL" in UPPERCASE
5. Only include document name and pasal, remove other words

Examples:
Input: "apa isi pasal 2 Undang Undang no 19 tahun 2003"
Output: "UU Nomor 19 TAHUN 2003 PASAL 2"

Input: "apa yang dijelaskan pada pasal 2 Undang Undang no 19 tahun 2003"
Output: "UU Nomor 19 TAHUN 2003 PASAL 2"

Input: "bunyi pasal 15 PP nomor 5 tahun 2020"
Output: "PP Nomor 5 TAHUN 2020 PASAL 15"

Input: "pasal 3 peraturan menteri bumn no 10 tahun 2018"
Output: "PERMEN BUMN Nomor 10 TAHUN 2018 PASAL 3"

User query: {user_query}

Respond with ONLY the reformed query in the exact format, nothing else."""

        response = self.client.chat.completions.create(
            model=self.reformer_model,
            messages=[{"role": "user", "content": reform_prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def search_direct_pasals(self, query: str) -> List[Tuple[str, float]]:
        """Search for pasal names using FAISS (for direct queries)"""
        # Generate embedding for the query
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in nama index
        distances, indices = self.nama_index.search(query_embedding, self.top_k_direct)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata_nama):
                continue
            
            similarity = 1 - dist
            
            try:
                pasal_id = self.metadata_nama[str(idx)]
                if similarity >= self.similarity_threshold:
                    results.append((pasal_id, similarity))
            except (KeyError, IndexError, TypeError):
                continue
        
        return results
    
    def reform_semantic_query(self, user_query: str) -> str:
        """Reform semantic query for better similarity search"""
        reform_prompt = f"""You are an expert at reforming user queries for legal document retrieval.

The user's query needs to be reformulated to better match legal article content in a vector database.

Original query: {user_query}

Reform this query to:
1. Use formal legal language
2. Focus on key legal concepts
3. Remove conversational elements
4. Keep the same meaning/intent
5. make it so that there the similarity search results is effective

e.g "buatkan 3 peraturan mengenai gaji minimum pekerja" -> you should reform this without the "buatkan 3 peraturan"
because there is no such text in legal documents, only focus on gaji minimum pekerja

Respond with ONLY the reformed query, nothing else."""

        response = self.client.chat.completions.create(
            model=self.reformer_model,
            messages=[{"role": "user", "content": reform_prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def search_similar_pasals(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Search for similar pasals using FAISS (for semantic queries)"""
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, self.top_k_semantic)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            similarity = 1 - dist
            
            try:
                pasal_id = self.metadata[str(idx)]
                if similarity >= self.similarity_threshold:
                    results.append((pasal_id, similarity))
            except (KeyError, IndexError, TypeError):
                continue
        
        return results
    
    def query_neo4j_semantic(self, pasal_id: str) -> Dict:
        """Query Neo4j for a specific pasal with full context"""
        pasal_id = str(pasal_id).strip()
        query = """
        MATCH (currentPasal:PASAL {Pasal_id: $pasal_id})
        OPTIONAL MATCH (parentBab:BAB)-[rel_bab_pasal:HAS_PASAL]->(currentPasal)
        OPTIONAL MATCH (parentDoc1:DOC)-[rel_doc_bab:HAS_BAB]->(parentBab)
        OPTIONAL MATCH (parentDoc2:DOC)-[rel_doc_pasal:HAS_PASAL]->(currentPasal)
        WITH 
          currentPasal,
          parentBab,
          COALESCE(parentDoc1, parentDoc2) AS parentDoc,
          rel_bab_pasal,
          rel_doc_bab,
          rel_doc_pasal
        OPTIONAL MATCH (currentPasal)-[rel_prev:PREV]->(previousPasal:PASAL)
        OPTIONAL MATCH (parentDoc)-[rel_dirubah:DIRUBAH_SEBAGIAN_OLEH]->(nextDoc:DOC)
        RETURN 
          currentPasal.Pasal AS Current_Pasal_Name,
          currentPasal.Content AS Current_Pasal_Content,
          previousPasal.Pasal AS Previous_Pasal_Name,
          previousPasal.Content AS Previous_Pasal_Content,
          parentBab.BAB AS Parent_BAB_Name,
          parentDoc.judul AS Parent_Doc_Title,
          nextDoc.judul AS Sebagian_Dirubah_Oleh
        """
        
        with self.driver.session() as session:
            result = session.run(query, pasal_id=pasal_id)
            record = result.single()
            return dict(record) if record else {}
    
    def humanize_response(self, user_query: str, neo4j_results: List[Dict]) -> str:
        """Generate human-friendly response using LLM"""
        if not neo4j_results or all(not result for result in neo4j_results):
            context_str = "NO RELEVANT INFORMATION FOUND"
        else:
            context_parts = []
            for i, result in enumerate(neo4j_results, 1):
                if result:
                    context_parts.append(f"\n--- Hasil {i} ---")
                    for key, value in result.items():
                        if value and not key.startswith('_'):
                            context_parts.append(f"{key}: {value}")
            context_str = "\n".join(context_parts)
        
        humanizer_prompt = f"""Anda adalah asisten hukum Indonesia yang membantu menjawab pertanyaan hukum.

Pertanyaan pengguna: {user_query}

Konteks dari basis pengetahuan:
{context_str}

Instruksi:
1. Jika konteks berisi "NO RELEVANT INFORMATION FOUND", jawab: "Maaf, basis pengetahuan saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."
3. Sebutkan pasal, bab, dan dokumen yang relevan
4. Gunakan bahasa yang formal tetapi mudah dipahami
5. Jika ada perubahan dokumen (Sebagian_Dirubah_Oleh), sebutkan informasinya
6. Jika pertanyaan tidak sesuai dengan topik ketenagakerjaan, dan Undang-undang. say that its out of scope
7. **IMPORTANT** Do not add aditional context other than the context given in the prompt
8. **CRITICAL** IF you think the context is not enough, say that you dont have the knowledge to answer the question

Jawaban Anda:"""

        response = self.client.chat.completions.create(
            model=self.humanizer_model,
            messages=[{"role": "user", "content": humanizer_prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    
    def process_query(self, user_query: str) -> str:
        """Main method to process user query"""
        # Step 1: Classify query
        query_type = self.classify_query(user_query)
        
        neo4j_results = []
        
        if query_type == "direct":
            # Reform direct query to standardized format
            reformed_query = self.reform_direct_query(user_query)
            
            # Search for direct pasal names using FAISS
            similar_pasals = self.search_direct_pasals(reformed_query)
            
            # Query Neo4j for each similar pasal
            for pasal_id, similarity in similar_pasals:
                result = self.query_neo4j_semantic(pasal_id)
                if result:
                    result['_similarity'] = similarity
                    neo4j_results.append(result)
        
        else:  # semantic
            # Reform query
            reformed_query = self.reform_semantic_query(user_query)
            
            # Embed query
            query_embedding = self.embed_query(reformed_query)
            
            # Search similar pasals
            similar_pasals = self.search_similar_pasals(query_embedding)
            
            # Query Neo4j for each similar pasal
            for pasal_id, similarity in similar_pasals:
                result = self.query_neo4j_semantic(pasal_id)
                if result:
                    result['_similarity'] = similarity
                    neo4j_results.append(result)
        
        # Humanize response
        return self.humanize_response(user_query, neo4j_results)
    
    def close(self):
        """Close all connections"""
        if hasattr(self, 'driver'):
            self.driver.close()


# Streamlit App
def main():
    st.set_page_config(
        page_title="Chatbot Hukum Indonesia",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Custom CSS - removed conflicting styles
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f4788;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            text-align: center;
            margin-bottom: 2rem;
            opacity: 0.8;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">⚖️ Chatbot Hukum Indonesia</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Asisten pintar untuk pertanyaan hukum Indonesia</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chatbot' not in st.session_state:
        with st.spinner("Menginisialisasi chatbot..."):
            try:
                st.session_state.chatbot = IndonesianLegalChatbot()
                st.success("✓ Chatbot siap digunakan!")
            except Exception as e:
                st.error(f"❌ Error menginisialisasi chatbot: {str(e)}")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Informasi")
        
        st.subheader("📝 Cara Menggunakan:")
        
        st.markdown("**Pertanyaan Langsung:**")
        st.code("Apa isi pasal 2 PP No 50 tahun 2012?")
        st.code("Bunyi pasal 166 UU No 13 tahun 2004")
        
        st.markdown("**Pertanyaan Semantik:**")
        st.code("Apa aturan untuk jaminan hari tua")
        st.code("siapa pemegang saham tertinggi di persero")
        
        st.divider()
        
        st.subheader("📚 Jenis Dokumen:")
        st.markdown("""
        - **UU** - Undang Undang
        - **PP** - Peraturan Pemerintah
        - **PERMEN BUMN** - Peraturan Menteri BUMN
        """)
        
        st.divider()
        
        st.subheader("📊 Statistik")
        st.metric("Total Pesan", len(st.session_state.messages))
        
        st.divider()
        
        if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    # Display info
    st.info("💡 **Tips:** Ajukan pertanyaan tentang peraturan dan undang-undang Indonesia. Bot ini dapat menjawab pertanyaan spesifik tentang pasal tertentu atau pertanyaan konseptual tentang hukum.")
    
    # Display chat messages using st.chat_message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Mencari informasi..."):
                try:
                    response = st.session_state.chatbot.process_query(prompt)
                    st.markdown(response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"❌ Maaf, terjadi kesalahan: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Footer
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem;">
            <p><strong>⚖️ Chatbot Hukum Indonesia</strong> | Powered by OpenAI & Neo4j</p>
            <p><em>Disclaimer: Informasi yang diberikan hanya untuk referensi. Konsultasikan dengan ahli hukum untuk keperluan legal formal.</em></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()