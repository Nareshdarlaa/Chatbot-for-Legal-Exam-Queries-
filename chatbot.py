#step 1: install requirments
!pip install -q scikit-learn

# Step 2: Import required packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Step 3: Define a small knowledge base (no CSV needed)
faq_data = [
    {
        "question": "What is the syllabus for CLAT 2025?",
        "answer": "CLAT 2025 syllabus includes: English Language, Current Affairs including General Knowledge, Legal Reasoning, Logical Reasoning, and Quantitative Techniques."
    },
    {
        "question": "How many questions are there in the English section?",
        "answer": "The English section typically contains 28-32 multiple choice questions."
    },
    {
        "question": "Give me last year’s cut-off for NLSIU Bangalore.",
        "answer": "The cut-off for NLSIU Bangalore last year was approximately 85-90 marks for General category."
    },
    {
        "question": "What is the marking scheme in CLAT?",
        "answer": "Each correct answer is awarded 1 mark, and 0.25 marks are deducted for every incorrect response."
    },
    {
        "question": "Is there negative marking in CLAT?",
        "answer": "Yes, there is a negative marking of 0.25 marks for each incorrect answer."
    },
    {
        "question": "How long is the CLAT exam?",
        "answer": "The CLAT exam duration is 2 hours (120 minutes)."
    },
    {
        "question": "Which colleges accept CLAT scores?",
        "answer": "22 National Law Universities (NLUs) in India accept CLAT scores, including NLSIU, NALSAR, and WBNUJS."
    },
    {
        "question": "When will CLAT 2025 be conducted?",
        "answer": "CLAT 2025 is tentatively scheduled for December 2024."
    },
    {
        "question": "How to prepare for Legal Reasoning in CLAT?",
        "answer": "Read newspapers, practice comprehension-based legal questions, and study basic legal concepts and current legal issues."
    },
    {
        "question": "How many total questions are in CLAT?",
        "answer": "CLAT consists of 120 multiple choice questions in total."
    }
]

#Step 4: Prepare the questions and answers
questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

#Step 5: Build the TF-IDF model
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

#Step 6: Define the chatbot function
def clat_chatbot(user_query):
    user_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_vector, question_vectors)
    best_match_idx = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_idx]

    if best_score < 0.2:
        return "Sorry, I couldn't find a relevant answer. Please try rephrasing your question."
    
    return answers[best_match_idx]

# Step 7: Chat loop
print("Welcome to the CLAT FAQ Chatbot!")
print("Type your question below (or type 'exit' to quit):\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye and best of luck for CLAT! 👋")
        break
    response = clat_chatbot(user_input)
    print("Bot:", response)
