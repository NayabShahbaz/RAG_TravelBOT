from sentence_transformers import SentenceTransformer
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import os
from google import genai
from google.genai import types
import re
from dotenv import load_dotenv

load_dotenv()

# --- Utility Functions (Kept as is) ---

def clean_value(val):
    if pd.isna(val):
        return "Unknown"
    return str(val)

def extract_numeric_price(price_str):
    """Extract numeric value from price string like 'Rs. 275,000'"""
    if pd.isna(price_str) or price_str == "Unknown" or price_str == "N/A":
        return float('inf')
    
    # Remove 'Rs.', commas, and spaces, then convert to float
    cleaned = str(price_str).replace('Rs.', '').replace(',', '').replace(' ', '')
    try:
        return float(cleaned)
    except ValueError:
        return float('inf')

def extract_numeric_duration(duration_str):
    """Extract number of days from duration string"""
    if pd.isna(duration_str) or duration_str == "Unknown" or duration_str == "N/A":
        return 0
    
    # Look for patterns like "5 Days", "7 Days 6 Nights"
    match = re.search(r'(\d+)\s*Days?', str(duration_str))
    if match:
        return int(match.group(1))
    return 0

def preprocess_tour_data(df):
    """Clean and preprocess tour data"""
    # Ensure Price is string and handle missing values
    df['Price'] = df['Price'].fillna('N/A').astype(str)
    df['Duration'] = df['Duration'].fillna('N/A').astype(str)
    df['Name'] = df['Name'].fillna('Unknown Tour')
    df['Itinerary'] = df['Itinerary'].fillna('No itinerary available')
    df['Link'] = df['Link'].fillna('No link available')
    
    # Handle Destination column (some CSVs might not have it)
    if 'Destination' not in df.columns:
        df['Destination'] = 'Unknown'
    else:
        df['Destination'] = df['Destination'].fillna('Unknown')
    
    return df

def load_all_tour_data():
    """Load and combine all tour data from multiple CSV files"""
    csv_files = [
        "data/hunza.csv",
        "data/naran.csv",
        "data/kumrat.csv", 
        "data/fairyMedows.csv",
        "data/murree.csv",
        "data/chitral.csv",
        "data/azadKashmir.csv",
        "data/neelum.csv",
        "data/swat.csv",
        "data/sakardu.csv"
    ]
    
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df = preprocess_tour_data(df)
            all_data.append(df)
            print(f"✓ Loaded {len(df)} tours from {csv_file.split('/')[-1]}")
        except Exception as e:
            print(f"✗ Could not load {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n🎯 Total tours loaded: {len(combined_df)}")
        print(f"🏔️ Destinations covered: {combined_df['Destination'].value_counts().to_dict()}")
        return combined_df
    else:
        print("❌ Error: No tour data could be loaded")
        return pd.DataFrame()

# --- 1. MODIFIED extract_query_parameters ---
def extract_query_parameters(query):
    """Extract destination, duration, and budget from query. Now extracts multiple destinations."""
    query_lower = query.lower()
    
    # Destinations list (Ensure all destination keywords are lower case)
    destinations = ['hunza', 'chitral', 'naran', 'kaghan', 'kumrat', 'neelum', 
                    'fairy meadows', 'murree', 'swat', 'skardu', 'kashmir']
    
    # Extract ALL destinations found in the query
    final_destinations = []
    for dest in destinations:
        if dest in query_lower:
            final_destinations.append(dest)
            
    # Extract duration
    duration_match = re.search(r'(\d+)\s*day', query_lower)
    target_days = int(duration_match.group(1)) if duration_match else None
    
    # Extract budget
    budget_match = re.search(r'(\d+,\d{3}|\d+)\s*(pkr|rs|rs\.|rupess?)', query_lower)
    if budget_match:
        max_price = float(budget_match.group(1).replace(',', ''))
    else:
        # Look for numbers that could be prices
        number_matches = re.findall(r'(\d+,\d{3}|\d{4,})', query)
        if number_matches:
            max_price = float(number_matches[0].replace(',', ''))
        else:
            max_price = None
    
    return final_destinations, target_days, max_price

# --- 2. MODIFIED filter_and_rank_tours ---
def filter_and_rank_tours(retrieved_docs, query, max_price=None, target_days=None, max_results=3):
    """
    Filter and rank tours based on query constraints.
    Returns the best tour per specified destination, or the top N overall.
    """
    destination_list, extracted_days, extracted_price = extract_query_parameters(query)
    
    # Use extracted parameters if available, otherwise use defaults
    if max_price is None:
        max_price = extracted_price if extracted_price else float('inf')
    if target_days is None:
        target_days = extracted_days if extracted_days else None
    
    # Dictionary to store the best tour found for each specific destination
    best_tours_by_destination = {}
    all_filtered_tours = [] # For general ranking if no destination specified
    
    # Normalize destination list for comparison
    target_destinations = [d.lower() for d in destination_list]
    
    for doc in retrieved_docs:
        price = extract_numeric_price(doc.get("Price", "N/A"))
        duration = extract_numeric_duration(doc.get("Duration", "N/A"))
        doc_destination = doc.get("Destination", "").lower()
        
        # Check destination match
        # If no destinations were specified, every document is a potential match.
        # If destinations WERE specified, the document must contain one of them.
        destination_match = True
        if target_destinations:
            destination_match = any(d in doc_destination for d in target_destinations)
        
        # Check if tour meets the criteria
        price_ok = price <= max_price if max_price != float('inf') else True
        duration_ok = (duration == target_days) if target_days else True
        
        # Calculate a score for ranking: lower is better (prioritize price, then duration match)
        score = price + (abs(duration - (target_days if target_days else 0)) * 5000)
        
        if price_ok and duration_ok and destination_match:
            tour_info = {
                'doc': doc,
                'price': price,
                'duration': duration,
                'score': score
            }

            # If a specific destination was targeted and found in the document
            matched_dest = next((d for d in target_destinations if d in doc_destination), None)
            
            if matched_dest:
                if matched_dest not in best_tours_by_destination or score < best_tours_by_destination[matched_dest]['score']:
                    best_tours_by_destination[matched_dest] = tour_info
            
            # If no destination was specified or if we are collecting all general matches
            if not target_destinations:
                all_filtered_tours.append(tour_info)


    # Final selection logic
    if best_tours_by_destination:
        # Return the best tour for each requested destination
        final_tours = [info['doc'] for info in best_tours_by_destination.values()]
        # Sort by best score overall (e.g., cheapest tour first)
        final_tours.sort(key=lambda doc: extract_numeric_price(doc.get("Price", "N/A")))
        return final_tours[:max_results]
    
    elif all_filtered_tours:
        # If no destination specified, return the top 'max_results' overall cheapest tours that match price/duration
        all_filtered_tours.sort(key=lambda x: x['score'])
        return [info['doc'] for info in all_filtered_tours][:max_results]
    
    # Fallback: If no *perfect* matches, find close alternatives
    # (The existing 'close match' logic is complex and often redundant with a good scoring function. 
    # For simplification, we'll revert to the top 3 results from the Pinecone retrieval if nothing
    # matched the strict price/duration filter.)
    
    # Fallback to the top 3 documents from the initial Pinecone retrieval, 
    # just ensuring they meet the destination filter if one was set.
    fallback_docs = []
    for doc in retrieved_docs:
        doc_destination = doc.get("Destination", "").lower()
        destination_match = True
        if target_destinations:
            destination_match = any(d in doc_destination for d in target_destinations)
        
        if destination_match:
            fallback_docs.append(doc)

    # If even fallback is used, prioritize the cheapest tour if price constraint was set
    if fallback_docs:
        # Sort fallback by price (cheaper first)
        fallback_docs.sort(key=lambda doc: extract_numeric_price(doc.get("Price", "N/A")))
        # Return the best one only, as a close match is often best presented as a single alternative
        return fallback_docs[:1]

    return []

# --- 3. MODIFIED generate_with_rag_input ---
def generate_with_rag_input(query, retrieved_docs, max_tokens=2048):
    """Generates a context-aware response for single or multiple tours."""

    if not retrieved_docs:
        return {"role": "assistant", "content": "No relevant tours found for your query."}

    # If only one tour is found (e.g., single destination query)
    if len(retrieved_docs) == 1:
        best_tour_metadata = retrieved_docs[0] 
        full_itinerary_text = best_tour_metadata.get("Itinerary", "N/A")
        extracted_highlights = get_clean_highlights(full_itinerary_text)
        
        final_output = f"""
I found the best match for your request! Here are the details for the tour:

**Tour Name:** {best_tour_metadata.get("Name", "N/A")}
**Destination:** {best_tour_metadata.get("Destination", "N/A")}
**Price:** {best_tour_metadata.get("Price", "N/A")}
**Duration:** {best_tour_metadata.get("Duration", "N/A")}
**Link:** {best_tour_metadata.get("Link", "N/A")}
**Highlights:**
{chr(10).join(extracted_highlights)}
"""
        return {
            "role": "assistant",
            "content": final_output.strip()
        }

    # If multiple tours are found (e.g., multiple destination query or general request)
    else:
        output_sections = [f"I found {len(retrieved_docs)} great options based on your request! Here are the best tours I could find:"]
        
        for i, tour in enumerate(retrieved_docs):
            full_itinerary_text = tour.get("Itinerary", "N/A")
            extracted_highlights = get_clean_highlights(full_itinerary_text)
            
            tour_section = f"""
### 🏞️ Option {i+1}: {tour.get("Name", "N/A")}

* **Destination:** {tour.get("Destination", "N/A")}
* **Price:** {tour.get("Price", "N/A")}
* **Duration:** {tour.get("Duration", "N/A")}
* **Link:** {tour.get("Link", "N/A")}

**Key Highlights:**
{chr(10).join(extracted_highlights)}
---
"""
            output_sections.append(tour_section.strip())
            
        final_output = "\n\n".join(output_sections)
        return {
            "role": "assistant",
            "content": final_output.strip()
        }

# --- Pinecone Setup (Kept as is for context) ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
INDEX_NAME = os.getenv('INDEX_NAME')
VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION'))
METADATA_LIMIT = int(os.getenv('METADATA_LIMIT'))

# Load all tour data
print("Loading all tour data...")
df = load_all_tour_data()

if df.empty:
    print("FATAL ERROR: No tour data loaded. Please check your CSV files.")
    exit()

# --- 1. Encoding Model ---
print("Initializing Sentence Transformer model...")
model_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. Pinecone Index Setup (Kept as is for context) ---
try:
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
        )
        print("Index created.")

    index = pc.Index(INDEX_NAME)
    print(f"Pinecone index connected. Status: {index.describe_index_stats()}")

    # Prepare data for upserting
    vectors_to_upsert = []
    for i, row in df.iterrows():
        text_to_embed = f"""
Name: {clean_value(row['Name'])}
Destination: {clean_value(row['Destination'])}
Duration: {clean_value(row['Duration'])}
Price: {clean_value(row['Price'])}
Itinerary: {clean_value(row['Itinerary'])}
"""
        embedding = model_embedder.encode(text_to_embed).tolist()
        
        vectors_to_upsert.append({
            "id": str(i),
            "values": embedding,
            "metadata": {
                "Name": clean_value(row["Name"]),
                "Destination": clean_value(row["Destination"]),
                "Duration": clean_value(row["Duration"]),
                "Price": clean_value(row["Price"]),
                "Link": clean_value(row["Link"]),
                "Itinerary": clean_value(row["Itinerary"]),
                "ShortItinerary": clean_value(row["Itinerary"])[:METADATA_LIMIT],
                "Text": text_to_embed
            }
        })
    
    print("Upserting vectors to Pinecone...")
    index.upsert(vectors=vectors_to_upsert)
    print("Upsert complete.")

except Exception as e:
    print(f"FATAL ERROR setting up Pinecone. Error details: {e}")
    exit()

# --- 3. Language Model Setup (Gemini API) (Kept as is for context) ---
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("\nWARNING: GEMINI_API_KEY environment variable not found. Please set it.")
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"\nGemini Client initialized successfully using model: {GEMINI_MODEL}.")
except Exception as e:
    print(f"FATAL ERROR initializing Gemini Client. Error: {e}")
    exit()

# --- Shared API Call Function (Kept as is) ---
def is_travel_query(query):
    q = query.lower()
    print(f"query:{q}")



    # Strong travel intent keywords (must be combined with a destination OR numbers)
    strong_intent = [
        "give me a", "i want a", "find me a",
        "show me a", "looking for", "plan a",
        "recommend", "suggest", "tour package", "trip package",
        "itinerary", "cheapest", "longest", "shortest", "available tours",  "which tours", "last","which tour", "tour"
    ]

    # Destinations list
    destination_keywords = [
        "hunza", "skardu", "swat", "naran", "kaghan", "gilgit", 
        "neelum", "kashmir", "murree", "chitral", 
        "fairy meadows", "kumrat", 'sightseeing', 'northern areas', 'northern pakistan', 'northern'
    ]

    # Check: does the query contain a destination?
    contains_destination = any(dest in q for dest in destination_keywords)

    # Check: does the query contain numbers (days or budget)?
    contains_number = bool(re.search(r"\d+", q))

    # Check strong intent phrases
    contains_strong_intent = any(phrase in q for phrase in strong_intent)

    # RULE: Trigger RAG only when:
    # (Strong intent AND destination) OR (destination AND numbers)
    if (contains_strong_intent or contains_destination) or \
       (contains_destination and contains_number):
        print("returning true")
        return True

    return False

def call_gemini_model(system_prompt, user_prompt, max_tokens):
    """Handles the communication with the Gemini API."""
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=max_tokens
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[user_prompt],
            config=config
        )
        if response.text:
            return response.text
        else:
            return "API_RETURNED_EMPTY_CONTENT"
    except Exception as e:
        return f"API_ERROR_FAILURE: {e}"

# --- 4. Generation Functions (Kept as is) ---

def get_clean_highlights(itinerary_text):
    """Uses Gemini to perform a simple, clean bullet-point extraction."""
    if not itinerary_text or itinerary_text.strip() == "" or itinerary_text == "Unknown":
        return ["• No highlights available"]

    # Improved prompt that works with both formats
    system_prompt_extractor = """You are a travel highlights extractor. Extract 3-5 key highlights from the tour itinerary. 
Focus on unique experiences, main attractions, and special features. 
Return ONLY bullet points starting with •, no other text.
Examples of good highlights:
• Continental breakfast included
• Flight from Islamabad to Gilgit
• Private transport throughout
• Views of Passu Cones
• Optional speed boat adventures at Attabad Lake
• Terrace and basecamp for hiking/trekking"""
    
    user_prompt_extractor = f"Extract 3-5 key highlights from this tour itinerary:\n{itinerary_text}"
    
    content = call_gemini_model(system_prompt_extractor, user_prompt_extractor, max_tokens=1000)

    if content.startswith("API_ERROR_FAILURE"):
        # Fallback: extract key phrases manually
        return extract_fallback_highlights(itinerary_text)

    # Clean and format the bullet points
    bullet_points = []
    for line in content.split('\n'):
        line = line.strip()
        if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
            # Clean the bullet point
            clean_line = line.lstrip('•-* ').strip()
            if clean_line:
                bullet_points.append(f"• {clean_line}")
    
    # If we got good bullet points, return them (limit to 5)
    if bullet_points and len(bullet_points) >= 2:
        return bullet_points[:5]
    else:
        # Fallback if Gemini didn't return proper bullet points
        return extract_fallback_highlights(itinerary_text)

def extract_fallback_highlights(itinerary_text):
    """Fallback method to extract highlights when Gemini fails"""
    highlights = []
    
    # Look for key features in the text
    text_lower = itinerary_text.lower()
    
    # Check for common tour features
    features_to_check = [
        "breakfast", "flight", "air ticket", "private transport", 
        "view", "lake", "fort", "valley", "glacier", "hiking", 
        "trekking", "boating", "bazar", "resort", "adventure"
    ]
    
    sentences = re.split(r'[.!?]', itinerary_text)
    for sentence in sentences:
        sentence = sentence.strip()
        if any(feature in sentence.lower() for feature in features_to_check) and len(sentence) > 10:
            # Shorten long sentences
            if len(sentence) > 80:
                words = sentence.split()[:12]  # Take first 12 words
                shortened = ' '.join(words) + '...'
            
                highlights.append(f"• {shortened}")
            else:
                highlights.append(f"• {sentence}")
            
            if len(highlights) >= 5:
                break
    
    # If no features found, create generic highlights
    if not highlights:
        highlights = [
            "• Comprehensive tour package",
            "• Experienced local guides", 
            "• Scenic mountain views",
            "• Cultural experiences",
            "• Comfortable accommodations"
        ]
    
    return highlights[:5]

def generate_without_rag_input(prompt, max_tokens=2048):
    """Generates a response using only the query (no context) via Gemini API."""
    # Constrained prompt to prevent hallucination
    system_prompt = "You are a highly constrained travel assistant. Your task is to answer the user query ONLY with general knowledge and a disclaimer about not having specific data. DO NOT invent budgets, specific tour names, or itineraries."
    content = call_gemini_model(system_prompt, prompt, max_tokens)
    return {"role": "assistant", "content": content}

def generate_non_travel_input(prompt, language='en', max_tokens=1000):
    lang_instruction="Answer in Urdu." if language=='ur'else "Answer in English."
    """
    Generates a helpful, natural, non-travel response using general reasoning only.
    No hallucinated facts, no invented data — only conversational assistance.
    """

    system_prompt = f"""
You are ViaNova, an AI assistant with a friendly, helpful personality.
{lang_instruction}
You can answer general user questions clearly and conversationally.

Rules:
- If the user asks about general knowledge, explain briefly and accurately.
- If the user asks something unknown or unverifiable, say you are not fully sure.
- Do NOT invent fake statistics, dates, medical claims, or technical details.
- Keep responses concise, helpful, and human-like.
- You are NOT restricted to travel — you can answer ANY safe topic.
"""

    user_prompt = f"User message: {prompt}\nProvide the best possible helpful reply."

    content = call_gemini_model(system_prompt, user_prompt, max_tokens)
    if(content.startswith("API_ERROR_FAILURE") or content == "API_RETURNED_EMPTY_CONTENT"):
        content = "Sorry, I'm having trouble processing your request right now."
    return {"role": "assistant", "content": content}


# --- 5. Interactive Query System ---

def process_user_query(query):
    """
    Process a single user query and return a dict:
    {
        'rag_output': <RAG answer or None>,
        'general_output': <non-RAG answer>,
        'model_answer': <best available answer>
    }
    """

    # --- 1. Check if this is a travel query ---
    if not is_travel_query(query):
        general_response = generate_non_travel_input(query)
        return {
            "rag_output": None,
            "general_output": general_response['content'],
            "model_answer": general_response['content']
        }

    # --- 2. Extract parameters and retrieve documents ---
    destination, target_days, max_price = extract_query_parameters(query)
    
    query_embedding = model_embedder.encode([query]).tolist()
    search_results = index.query(vector=query_embedding, top_k=15, include_metadata=True)
    retrieved_docs = [match['metadata'] for match in search_results['matches']]

    filtered_docs = filter_and_rank_tours(retrieved_docs, query, max_price, target_days, max_results=3)

    # --- 3. Generate outputs ---
    if filtered_docs:
        rag_output = generate_with_rag_input(query, filtered_docs)['content']
        general_output = generate_without_rag_input(query)['content']

        # Return a dict that app.py expects
        return {
            "rag_output": rag_output,
            "general_output": general_output,
            "model_answer": rag_output or general_output
        }

    else:
        # No tours matched, only general output
        general_output = generate_without_rag_input(query)['content']
        return {
            "rag_output": None,
            "general_output": general_output,
            "model_answer": general_output
        }