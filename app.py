import streamlit as st
import pandas as pd
import torch
import os
from datetime import datetime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Configuration
load_dotenv('./.env')


QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# Load models
@st.cache_resource
def load_models():
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    return qdrant_client, embedder, tokenizer, model

qdrant_client, embedder, tokenizer, model = load_models()

# RAG Pipeline
def ask_recipe(question, max_steps_tokens=400):
    query_vector = embedder.encode(question).tolist()
    results = qdrant_client.query_points(collection_name="recipes", query=query_vector, limit=1)
    
    recipe = results.points[0].payload
    recipe_id = results.points[0].id
    name = recipe.get('name', 'Unknown Recipe')
    ingredients = recipe.get('ingredients', 'Not available')
    instructions = recipe.get('combined_text_clean', '')
    
    if len(instructions) > 400:
        instructions = instructions[:400] + "..."
    
    prompt = f"""<|system|>
You are a helpful recipe assistant.</s>
<|user|>
Recipe: {name}
Instructions: {instructions}

Provide clear numbered cooking steps.</s>
<|assistant|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_steps_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    steps = full_response.split('<|assistant|>')[-1].strip() if '<|assistant|>' in full_response else full_response[len(prompt):].strip()
    
    answer = f"**Recipe:** {name}\n\n**Ingredients:** {ingredients}\n\n**Steps:**\n{steps}"
    return answer, recipe_id, name

# Save user feedback
def save_feedback(question, recipe_name, rating):
    data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'question': [question],
        'recipe_name': [recipe_name],
        'rating': [rating]
    }
    df = pd.DataFrame(data)
    
    if os.path.exists('feedback.csv'):
        df.to_csv('feedback.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('feedback.csv', mode='w', header=True, index=False)

# Streamlit UI
st.set_page_config(page_title="Recipe RAG", page_icon="🍝", layout="wide")
st.title("👩🏻‍🍳 Personal Food Recipe Assistant 🧑🏻‍🍳")

page = st.sidebar.radio("Navigate", ["Search Recipe", "Dashboard"])

if page == "Search Recipe":
    st.write("Hello!😃 Craving something delicious? Let's find your recipe. 🍲 😋")
    
    question = st.text_input("What would you like to cook?")
    
    if st.button("Get Recipe", type="primary"):
        if question:
            with st.spinner("Searching and generating recipe..."):
                try:
                    answer, recipe_id, recipe_name = ask_recipe(question)
                    
                    st.session_state['last_question'] = question
                    st.session_state['last_recipe'] = recipe_name
                    st.session_state['show_feedback'] = True
                    
                    st.markdown("---")
                    st.markdown(answer)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.session_state.get('show_feedback', False):
        st.markdown("---")
        st.subheader("Was this helpful?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👍 Yes, helpful!"):
                save_feedback(
                    st.session_state['last_question'],
                    st.session_state['last_recipe'],
                    "positive"
                )
                st.success("Thank you for your feedback!")
        
        with col2:
            if st.button("👎 Not helpful"):
                save_feedback(
                    st.session_state['last_question'],
                    st.session_state['last_recipe'],
                    "negative"
                )
                st.info("Thanks for your feedback!")

else:
    st.header("📊 Monitoring Dashboard")
    
    if not os.path.exists('feedback.csv'):
        st.info("No data yet. Search recipes and provide feedback to see metrics here.")
    else:
        try:
            df = pd.read_csv('feedback.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Queries", len(df))
            col2.metric("Unique Recipes", df['recipe_name'].nunique())
            
            positive = len(df[df['rating'] == 'positive'])
            satisfaction = (positive / len(df) * 100) if len(df) > 0 else 0
            col3.metric("Satisfaction Rate", f"{satisfaction:.0f}%")
            
            st.markdown("---")
            
            # Chart 1: Queries Over Time
            st.subheader("1. Queries Over Time")
            daily = df.groupby('date').size()
            st.line_chart(daily)
            
            # Chart 2: Popular Recipes
            st.subheader("2. Most Popular Recipes")
            top_recipes = df['recipe_name'].value_counts().head(10)
            st.bar_chart(top_recipes)
            
            # Chart 3: Feedback Distribution
            st.subheader("3. User Feedback Distribution")
            feedback = df['rating'].value_counts()
            st.bar_chart(feedback)
            
            # Chart 4: Recent Queries
            st.subheader("4. Recent Queries")
            st.dataframe(df[['timestamp', 'question', 'recipe_name', 'rating']].tail(10), use_container_width=True)
            
            # Chart 5: Common Questions
            st.subheader("5. Most Common Questions")
            questions = df['question'].value_counts().head(5)
            st.bar_chart(questions)
            
            # Chart 6: Activity by Hour
            st.subheader("6. Activity by Hour of Day")
            hourly = df['hour'].value_counts().sort_index()
            st.bar_chart(hourly)
            
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")