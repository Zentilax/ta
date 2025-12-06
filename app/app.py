import os
import json
import faiss
import numpy as np
from openai import OpenAI
from neo4j import GraphDatabase
from typing import List, Dict, Tuple, Optional
import re

class IndonesianLegalChatbot:
    def __init__(self):
        """Initialize the legal chatbot with all necessary connections and data"""
        print("\n" + "="*60)
        print("INITIALIZING INDONESIAN LEGAL CHATBOT")
        print("="*60)
        
        # Initialize OpenAI client
        print("\n[1/4] Connecting to OpenAI API...")
        self.client = OpenAI(api_key="")
        print("✓ OpenAI API connected")
        
        # Load FAISS index and metadata
        print("\n[2/4] Loading FAISS index and metadata...")
        self.faiss_index_path = r"C:\Users\Aqil Anhein\Desktop\TA\app\pasal.index"
        self.metadata_path = r"C:\Users\Aqil Anhein\Desktop\TA\app\id_mapping.json"
        
        try:
            self.index = faiss.read_index(self.faiss_index_path)
            print(f"✓ FAISS index loaded: {self.index.ntotal} vectors")
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"✓ Metadata loaded: {len(self.metadata)} entries")
        except Exception as e:
            print(f"✗ Error loading FAISS/metadata: {e}")
            raise
        
        # Connect to Neo4j
        print("\n[3/4] Connecting to Neo4j...")
        self.neo4j_uri = "bolt://localhost:7687"  # CHANGE THIS
        self.neo4j_user = "neo4j"  # CHANGE THIS
        self.neo4j_password = "Ghazali12."  # CHANGE THIS
        
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            print("✓ Neo4j connected successfully")
        except Exception as e:
            print(f"✗ Neo4j connection failed: {e}")
            raise
        
        # Configuration
        print("\n[4/4] Setting up configuration...")
        self.top_k = 3
        self.similarity_threshold = 0.0001
        self.classifier_model = "gpt-4o-mini"
        self.reformer_model = "gpt-4o-mini"
        self.humanizer_model = "gpt-4o-mini"  # CHANGE TO gpt-4o when available
        self.embedding_model = "text-embedding-3-small"
        print("✓ Configuration loaded")
        print(f"  - Top K: {self.top_k}")
        print(f"  - Similarity threshold: {self.similarity_threshold}")
        print(f"  - Classifier model: {self.classifier_model}")
        print(f"  - Humanizer model: {self.humanizer_model}")
        
        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60 + "\n")
    
    def classify_query(self, user_query: str) -> str:
        """Classify if query is 'direct' or 'semantic'"""
        print("\n" + "-"*60)
        print("STEP 1: CLASSIFYING QUERY TYPE")
        print("-"*60)
        print(f"User query: {user_query}")
        
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

        try:
            response = self.client.chat.completions.create(
                model=self.classifier_model,
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0
            )
            classification = response.choices[0].message.content.strip().lower()
            print(f"Classification result: {classification.upper()}")
            return classification
        except Exception as e:
            print(f"✗ Classification error: {e}")
            return "semantic"  # Default to semantic on error
    
    def generate_cypher_query_direct(self, user_query: str) -> Tuple[Optional[str], Optional[str]]:
        """Use LLM to generate Cypher query for direct questions"""
        print("\n" + "-"*60)
        print("STEP 2A: GENERATING CYPHER QUERY (DIRECT)")
        print("-"*60)
        print(f"User query: {user_query}")
        
        cypher_prompt = f"""You are a Cypher query generator for an Indonesian legal knowledge graph.

Database Schema:
- DOC nodes: {{judul, judul_id, Label}}
- BAB nodes: {{BAB, BAB_id, Label}}
- PASAL nodes: {{Pasal, Pasal_id, Content, Label}}
- Relationships: DOC-[:HAS_BAB]->BAB, BAB-[:HAS_PASAL]->PASAL, DOC-[:HAS_PASAL]->PASAL, PASAL-[:PREV]->PASAL, DOC-[:DIRUBAH_SEBAGIAN_OLEH]->DOC

ID Format Rules:
1. Doc type codes: UU=1, Peraturan Menteri=2, PP=3, PERPPU=4
2. Document ID (judul_id): [1 digit type][4 digit nomor][2 digit year]
   Example: UU Nomor 70 Tahun 2004 = 1007504
   Example : PP Nomor 4 Tahun 2003 = 3000403
   Example : PP Nomor 144 Tahun 2003 = 3014403
3. BAB ID (BAB_id): [doc_id][2 digit bab number]
   Example: Bab 3 in above doc = 101230403
4. PASAL ID (Pasal_id): [doc_id][2 digit bab][3 digit pasal]
   Example: Pasal 15 Bab 3 in above doc = 101230403015
   Example: Pasal 1 (no bab) in above doc = 101230400001

User Query: {user_query}

Generate a Cypher query to retrieve the requested information. The query should:
1. Match the relevant nodes based on the IDs
2. Return all related context (parent BAB, parent DOC, previous PASAL, document changes)
3. Use OPTIONAL MATCH for relationships that might not exist

IMPORTANT: 
- Year format: 2004 -> 04, 1999 -> 99, 2000 -> 00
- Always pad numbers with zeros to correct length

Respond with ONLY the Cypher query, no explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.classifier_model,
                messages=[{"role": "user", "content": cypher_prompt}],
                temperature=0
            )
            cypher_query = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if cypher_query.startswith("```"):
                cypher_query = cypher_query.split("```")[1]
                if cypher_query.startswith("cypher"):
                    cypher_query = cypher_query[6:]
                cypher_query = cypher_query.strip()
            
            print(f"\nGenerated Cypher (WITH BAB):\n{cypher_query}")
            
            # Generate alternative query without BAB
            print("\nGenerating alternative query without BAB structure...")
            alt_prompt = f"""Modify the following Cypher query to work for documents WITHOUT BAB structure.
 
This means:
1. Change PASAL IDs to have '00' for the BAB portion
2. Remove BAB-related MATCH clauses
3. Only use DOC-[:HAS_PASAL]->PASAL relationships

ID Format Rules:
1. Doc type codes: UU=1, Peraturan Menteri=2, PP=3
2. Document ID (judul_id): [1 digit type][4 digit nomor][2 digit year]
   Example: UU No 123 Tahun 2004 = 1012304
3. BAB ID (BAB_id): [doc_id][2 digit bab number]
   Example: Bab 3 in above doc = 101230403
4. PASAL ID (Pasal_id): [doc_id][2 digit bab][3 digit pasal]
   Example: Pasal 15 Bab 3 in above doc = 101230403015
   Example: Pasal 1 (no bab) in above doc = 101230400001  

Original query:
{cypher_query}

Respond with ONLY the modified Cypher query."""

            alt_response = self.client.chat.completions.create(
                model=self.classifier_model,
                messages=[{"role": "user", "content": alt_prompt}],
                temperature=0
            )
            alt_cypher_query = alt_response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if alt_cypher_query.startswith("```"):
                alt_cypher_query = alt_cypher_query.split("```")[1]
                if alt_cypher_query.startswith("cypher"):
                    alt_cypher_query = alt_cypher_query[6:]
                alt_cypher_query = alt_cypher_query.strip()
            
            print(f"\nGenerated Cypher (WITHOUT BAB):\n{alt_cypher_query}")
            
            return cypher_query, alt_cypher_query
            
        except Exception as e:
            print(f"✗ Cypher generation error: {e}")
            return None, None
    
    def execute_cypher_query(self, cypher_query: str, alt_cypher_query: Optional[str] = None) -> List[Dict]:
        """Execute Cypher query on Neo4j, try alternative if first fails"""
        print("\n" + "-"*60)
        print("STEP 3A: EXECUTING CYPHER QUERY")
        print("-"*60)
        
        results = []
        
        with self.driver.session() as session:
            # Try first query
            print("\nAttempt 1: Executing query WITH BAB structure...")
            try:
                result = session.run(cypher_query)
                records = list(result)
                
                if records and any(record.values() for record in records):
                    print(f"✓ Found {len(records)} record(s)")
                    for record in records:
                        results.append(dict(record))
                else:
                    print("✗ No results found with BAB structure")
                    
                    # Try alternative query without BAB
                    if alt_cypher_query:
                        print("\nAttempt 2: Executing query WITHOUT BAB structure...")
                        result = session.run(alt_cypher_query)
                        records = list(result)
                        
                        if records and any(record.values() for record in records):
                            print(f"✓ Found {len(records)} record(s)")
                            for record in records:
                                results.append(dict(record))
                        else:
                            print("✗ No results found without BAB structure")
                    
            except Exception as e:
                print(f"✗ Query execution error: {e}")
                
                # Try alternative query if available
                if alt_cypher_query:
                    print("\nAttempt 2: Trying alternative query...")
                    try:
                        result = session.run(alt_cypher_query)
                        records = list(result)
                        
                        if records:
                            print(f"✓ Found {len(records)} record(s)")
                            for record in records:
                                results.append(dict(record))
                        else:
                            print("✗ No results found")
                    except Exception as e2:
                        print(f"✗ Alternative query also failed: {e2}")
        
        return results
    
    def reform_semantic_query(self, user_query: str) -> str:
        """Reform semantic query for better similarity search"""
        print("\n" + "-"*60)
        print("STEP 2B: REFORMING SEMANTIC QUERY")
        print("-"*60)
        print(f"Original query: {user_query}")
        
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

        try:
            response = self.client.chat.completions.create(
                model=self.reformer_model,
                messages=[{"role": "user", "content": reform_prompt}],
                temperature=0.3
            )
            reformed_query = response.choices[0].message.content.strip()
            print(f"Reformed query: {reformed_query}")
            return reformed_query
        except Exception as e:
            print(f"✗ Reform error: {e}, using original query")
            return user_query
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        print("\n" + "-"*60)
        print("STEP 3B: GENERATING QUERY EMBEDDING")
        print("-"*60)
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            print(f"✓ Embedding generated: dimension {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"✗ Embedding error: {e}")
            raise
    
    def search_similar_pasals(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Search for similar pasals using FAISS"""
        print("\n" + "-"*60)
        print("STEP 4B: SEARCHING SIMILAR PASALS")
        print("-"*60)
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        print(f"Top {self.top_k} results:")
        print(f"FAISS index size: {self.index.ntotal}")
        print(f"Metadata size: {len(self.metadata)}")
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            print(f"\n  [{i+1}] FAISS Index: {idx} | Distance: {dist:.4f}")
            
            # Check if index is valid
            if idx < 0 or idx >= len(self.metadata):
                print(f"      ✗ Invalid index {idx} (metadata size: {len(self.metadata)})")
                continue
            
            # Convert distance to similarity (assuming cosine distance)
            similarity = 1 - dist
            
            # Get pasal_id from metadata
            try:
                pasal_id = self.metadata[str(idx)]
                print(f"      Pasal ID: {pasal_id} | Similarity: {similarity:.4f}")
                
                if similarity >= self.similarity_threshold:
                    results.append((pasal_id, similarity))
                    print(f"      ✓ Above threshold ({self.similarity_threshold})")
                else:
                    print(f"      ✗ Below threshold ({self.similarity_threshold})")
            except (KeyError, IndexError, TypeError) as e:
                print(f"      ✗ Error accessing metadata at index {idx}: {e}")
                print(f"      Metadata type: {type(self.metadata)}")
                if isinstance(self.metadata, list) and len(self.metadata) > 0:
                    print(f"      Sample metadata entry: {self.metadata[0]}")
                continue
        
        if not results:
            print(f"\n✗ No results above threshold {self.similarity_threshold}")
        else:
            print(f"\n✓ {len(results)} result(s) above threshold")
        
        return results
    
    def query_neo4j_semantic(self, pasal_id: str) -> Dict:
        """Query Neo4j for a specific pasal with full context"""
        print(f"\nQuerying Neo4j for Pasal ID: {pasal_id}")
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
            
            if record:
                print(f"  ✓ Found context for {pasal_id}")
                return dict(record)
            else:
                print(f"  ✗ No context found for {pasal_id}")
                return {}
    
    def humanize_response(self, user_query: str, neo4j_results: List[Dict]) -> str:
        """Generate human-friendly response using LLM"""
        print("\n" + "-"*60)
        print("STEP 5: HUMANIZING RESPONSE")
        print("-"*60)
        
        if not neo4j_results or all(not result for result in neo4j_results):
            print("No context available - knowledge base insufficient")
            context_str = "NO RELEVANT INFORMATION FOUND"
        else:
            print(f"Processing {len(neo4j_results)} result(s)")
            context_parts = []
            for i, result in enumerate(neo4j_results, 1):
                if result:
                    context_parts.append(f"\n--- Hasil {i} ---")
                    for key, value in result.items():
                        if value:
                            context_parts.append(f"{key}: {value}")
            context_str = "\n".join(context_parts)
        
        humanizer_prompt = f"""Anda adalah asisten hukum Indonesia yang membantu menjawab pertanyaan hukum.

Pertanyaan pengguna: {user_query}

Konteks dari basis pengetahuan:
{context_str}

Instruksi:
1. Jika konteks berisi "NO RELEVANT INFORMATION FOUND", jawab: "Maaf, basis pengetahuan saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."
2. Jika ada konteks, jawab pertanyaan dengan jelas dan lengkap berdasarkan konteks yang diberikan
3. Sebutkan pasal, bab, dan dokumen yang relevan
4. Gunakan bahasa yang formal tetapi mudah dipahami
5. Jika ada perubahan dokumen (Sebagian_Dirubah_Oleh), sebutkan informasinya

Jawaban Anda:"""

        try:
            response = self.client.chat.completions.create(
                model=self.humanizer_model,
                messages=[{"role": "user", "content": humanizer_prompt}],
                temperature=0.5
            )
            final_answer = response.choices[0].message.content.strip()
            print("✓ Response generated")
            return final_answer
        except Exception as e:
            print(f"✗ Humanizer error: {e}")
            return "Maaf, terjadi kesalahan dalam menghasilkan jawaban."
    
    def process_query(self, user_query: str) -> str:
        """Main method to process user query"""
        print("\n" + "="*60)
        print("PROCESSING NEW QUERY")
        print("="*60)
        
        # Step 1: Classify query
        query_type = self.classify_query(user_query)
        
        neo4j_results = []
        
        if query_type == "direct":
            # Step 2A: Generate Cypher queries using LLM
            cypher_query, alt_cypher_query = self.generate_cypher_query_direct(user_query)
            
            if cypher_query:
                # Step 3A: Execute Cypher queries on Neo4j
                neo4j_results = self.execute_cypher_query(cypher_query, alt_cypher_query)
            else:
                print("\n✗ Could not generate Cypher query")
        
        else:  # semantic
            # Step 2B: Reform query
            reformed_query = self.reform_semantic_query(user_query)
            
            # Step 3B: Embed query
            query_embedding = self.embed_query(reformed_query)
            
            # Step 4B: Search similar pasals
            similar_pasals = self.search_similar_pasals(query_embedding)
            
            # Step 5B: Query Neo4j for each similar pasal
            print("\n" + "-"*60)
            print("STEP 5B: QUERYING NEO4J FOR SIMILAR PASALS")
            print("-"*60)
            
            for pasal_id, similarity in similar_pasals:
                result = self.query_neo4j_semantic(pasal_id)
                if result:
                    result['_similarity'] = similarity
                    neo4j_results.append(result)
        
        # Step 6: Humanize response
        final_answer = self.humanize_response(user_query, neo4j_results)
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        
        return final_answer
    
    def close(self):
        """Close all connections"""
        print("\nClosing connections...")
        if hasattr(self, 'driver'):
            self.driver.close()
            print("✓ Neo4j connection closed")


def main():
    """Main function to run the chatbot"""
    try:
        # Initialize chatbot
        chatbot = IndonesianLegalChatbot()
        
        print("\n" + "="*60)
        print("INDONESIAN LEGAL CHATBOT")
        print("="*60)
        print("Ketik 'exit' atau 'quit' untuk keluar")
        print("="*60 + "\n")
        
        while True:
            # Get user input
            user_input = input("\n👤 Anda: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'keluar']:
                print("\nTerima kasih telah menggunakan chatbot hukum Indonesia!")
                break
            
            # Process query
            try:
                answer = chatbot.process_query(user_input)
                print("\n" + "="*60)
                print("🤖 JAWABAN:")
                print("="*60)
                print(answer)
                print("="*60)
            except Exception as e:
                print(f"\n✗ Error processing query: {e}")
        
        # Cleanup
        chatbot.close()
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()