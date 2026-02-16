import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GEMINI_API_KEY")

def initialize_llm(api_key):
    genai.configure(api_key = my_api_key)
    model = genai.GenerativeModel("gemini-flash-latest")
    return model

def build_prompt(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)       #joining all the chunks together with line breaks
    prompt = f"""You are a helpful AI Assistant that answers questions based on provided context.
    Context : {context}
    Question : {query}
    Instruction:
    Answer based only on the context above 
    If the answer is not in the context, say "I cannot find information in the provided context."
    Be concise and clear
    Answer : """
    return prompt

def generate_answer(query, relevant_chunks, api_key):
    try:
        model = initialize_llm(api_key)
        prompt = build_prompt(query, relevant_chunks)
        response = model.generate_content(prompt)
        answer = response.text
        return answer
    except Exception as e:
        return f"Error : {str(e)}"

#testing
if __name__ == "__main__":
    test_chunks = [
        "The Eiffel Tower is in Paris, France.",
        "It was built in 1889.",
        "Gustave Eiffel designed it."
    ]
    
    test_query = "Where is the Eiffel Tower?"
    
    print(f"Question: {test_query}\n")
    answer = generate_answer(test_query, test_chunks, my_api_key)
    print(f"Answer: {answer}")