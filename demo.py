import ollama
import sys
import math

#stores the (chunk, embedding) tuple
VECTOR_DB = []

#given in the exercise repo
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
#given in the exercise repo
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"


def cosine_similarity(a, b):
    """returns the cosine similarity between two vectors a and b"""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    norm_a = math.sqrt(sum(x*x for x in a))
    
    if norm_a == 0.0:
        return 0.0
    
    norm_b = math.sqrt(sum(y*y for y in b))
    
    if norm_b == 0.0:
        return 0.0

    dot_product = sum(x*y for x, y in zip(a, b))
        
    return dot_product / (norm_a*norm_b)

def add_chunk_to_database(chunk):
    """Embedding will be requested and appendedto VECTOR_DB 
       in the form of (chunk, embedding)"""

    try:
        response = ollama.embed(model=EMBEDDING_MODEL, 
                                input=chunk)
        
        embedding = response["embeddings"][0]

        VECTOR_DB.append((chunk, 
                                embedding))
    except Exception as e:
        print(f"Error embedding chunk: {e}")
        sys.exit(1)

def load_knowledge_base(file_path):
    """After loading the text file,
       lines will be created to chunk and addeded to VECTOR_DB
       via add_chunk_to_database()"""

    global VECTOR_DB
    VECTOR_DB = [] 

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            ds = [line.strip() for line in file if line.strip()]
        for chunk in ds:
            add_chunk_to_database(chunk)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'.")
        sys.exit(1)

def retrieve(query, top_n=3):
    """Query gets embedded and the TOP_N most similar chunks are retrieved.
       N is set to 3 as default"""
    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query,
    )["embeddings"][0]

    similarities = []
    for chunk, embedding in VECTOR_DB:
        score = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, score))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]

def write_query(input_query, top_n=3):
    """runs, formats and generates"""

    retrieved_knowledge = retrieve(input_query, 
                        top_n=top_n)

    print("\nRetrieved knowledge:")

    for chunk, similarity in retrieved_knowledge:
        print(f"- ({similarity:.3f}) {chunk}")

    context = "\n".join(
        f"- {chunk}" for chunk, _similarity in retrieved_knowledge
    )

    # Construct Grounded Prompt
    instruction_prompt = f"""You are a grounded question-answering assistant.
Use only the context below to answer the user's question.
If the context does not contain enough evidence, say that the answer is not in the knowledge base.
When records conflict, prefer a clearly dated newer record and explain the update briefly.

Context:
{context}
"""

    print("\nAnswer:")
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": input_query},
        ],
        stream=True,
    )

    for response_chunk in stream:
        print(response_chunk["message"]["content"], end="", flush=True)
    print("\n")

def _test_cosine_similarity():
    assert abs(cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine_similarity([1, 0], [0, 1]) - 0.0) < 1e-9
    assert abs(cosine_similarity([1, 1], [-1, -1]) - (-1.0)) < 1e-9
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    print("cosine_similarity tests passed")

def main():
    print("Hello, this is a RAG System that knows cats!!")
    
    knowledge_base_file= input("Please enter knowledge base file you want to load:").strip()
    if not knowledge_base_file:
        knowledge_base_file = "cat-facts.txt"
        
    load_knowledge_base(knowledge_base_file)
    
    top_n_input = input("Please enter the retrieval amount for top_n:").strip()
    
    top_n = int(top_n_input) if top_n_input.isdigit() else 3
    
    while True:
        try:
            input_query = input("\nPlease ask your question, or type 'exit' or 'quit' to quit:").strip()
            if input_query.lower() in ["exit", "quit"]:
                break
            if not input_query:
                continue
            
            write_query(input_query, top_n=top_n)
        except KeyboardInterrupt:
            break
            
if __name__ == "__main__":
    _test_cosine_similarity()
    main()