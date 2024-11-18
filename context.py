from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Comprehensive Academic Calendar Context
context_str = '''
January 6, 2025
- Classes in S courses begin
- Classes in Y courses resume

January 14, 2025
- Waitlists for S courses close at end of day

January 19, 2025
- Last day to enrol in S courses
- Program/Course fee freeze date (S courses). Any S courses cancelled after this date will still be counted for accurate tuition assessment. 

January 20, 2025
- First day to select a Credit/No-Credit (CR/NCR) option for S courses

January 31, 2025
- Last day to request Spring 2025 graduation

February 14, 2025
- Last day to drop Y courses
Note: Some courses cannot be dropped using ACORN and students must contact their College or Department to do so

February 17, 2025
- Family Day - University closed; no classes

February 17 - 21, 2025
- Reading Week

TBA
- Deferred exam period used for students who missed a final exam in December 2024
- Deadline to report a conflict for April 2024 final exams

March 10, 2025
- Last day to drop S courses
Note: Some courses cannot be dropped using ACORN and students must contact their College or Department to do so

April 4, 2025
- Classes end in S and Y courses
- Last day to add or remove a CR/NCR option in S and Y courses
- Deadline to request Late Withdrawal (LWD) from S and Y courses at College Registrar's Office

April 7–8, 2025
- Study days

April 9–30, 2025
- Final exams in S and Y courses

April 18, 2025
- Good Friday - University closed; no classes or final exams

TBA
- Check ACORN for results of the first request period to enrol in limited programs

May 7, 2025
- Last day to submit a term work extension petition in S & Y courses
- Last day for instructors to accept late term work for S or Y courses
'''

# Enhanced Prompt Template with Clear Instructions
template = """
CONTEXT PROCESSING INSTRUCTIONS:
- Carefully analyze the entire academic calendar context
- Extract precise, relevant information for the question
- Provide specific dates and details
- If no exact match exists, state "Information not found"
- Prioritize completeness and accuracy

ACADEMIC CALENDAR CONTEXT:
{context}

SPECIFIC QUESTION: {question}

PRECISE ANSWER:"""

# Create Prompt Template
prompt = ChatPromptTemplate.from_template(template)

# Configure Ollama Model with Low Temperature for Consistency
model = ChatOllama(
    model="llama3.2",  # Smaller model for faster processing
    temperature=0.1       # Low temperature to reduce randomness
)

# Output Parser for Structured Response
output_parser = StrOutputParser()

# Comprehensive Chain with Context Handling
chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

# Example Queries to Test Context Understanding
queries = [
    "When is drop deadline for Y courses?",
    "What are the key dates in February?",
    "When do classes end?",
    "What is the last day to submit term work extension?"
]

# Run Queries and Print Responses
for query in queries:
    print(f"\nQuery: {query}")
    response = chain.invoke({
        "question": query,
        "context": context_str
    })
    print(f"Response: {response}")