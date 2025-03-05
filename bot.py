import asyncio
import openai
import pandas as pd
import ast
import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import redis
import azure.cognitiveservices.speech as speechsdk
import nest_asyncio
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Nest async applied.")
nest_asyncio.apply()

# Azure Cognitive Search imports
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
logging.info("Azure Search modules imported.")

# Bot Framework dependencies
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes
logging.info("Bot Framework modules imported.")

# RT client for GPT‑4o realtime fallback (make sure the rtclient package is installed)
from rtclient import RTLowLevelClient, ResponseCreateMessage, ResponseCreateParams
logging.info("RT client modules imported.")

# ------------------------------------------------------------------
# Configuration for Azure OpenAI, GPT‑4o realtime, Azure Search, Redis, Speech
# ------------------------------------------------------------------
logging.info("Setting up configuration...")

# Azure OpenAI configuration for embeddings
OPENAI_API_KEY = "8929107a6a6b4f37b293a0fa0584ffc3"
OPENAI_API_VERSION = "2023-03-15-preview"
OPENAI_ENDPOINT = "https://genral-openai.openai.azure.com/"
EMBEDDING_MODEL = "text-embedding-ada-002"  # Fast embedding model

# GPT‑4o realtime 
RT_API_KEY = "9e76306d48fb4e6684e4094d217695ac"
RT_ENDPOINT = "https://general-openai02.openai.azure.com/"
RT_DEPLOYMENT = "gpt-4o-realtime-preview"
RT_API_VERSION = "2024-10-17"

# Azure Cognitive Search 
SEARCH_SERVICE_NAME = "mainsearch01"          
SEARCH_INDEX_NAME = "id"                      
SEARCH_API_KEY = "Y6dbb3ljV5z33htXQEMR8ICM8nAHxOpNLwEPwKwKB9AzSeBtGPav"

# Redis 
REDIS_HOST = "AiKr.redis.cache.windows.net"
REDIS_PORT = 6380
REDIS_PASSWORD = "OD8wyo8NiVxse6DDkEY19481Xr7ZhQAnfAzCaOZKR2U="

# Speech 
SPEECH_KEY = "3c358ec45fdc4e6daeecb7a30002a9df"
SPEECH_REGION = "westus2"

# Thresholds for determining whether a search result is “good enough.”
SEMANTIC_THRESHOLD = 3.4 
VECTOR_THRESHOLD = 0.91

logging.info("Configuration constants set.")

# ------------------------------------------------------------------
# Initialize clients and load Q&A data
# ------------------------------------------------------------------
logging.info("Initializing Azure OpenAI client for embeddings...")
client = openai.AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)
logging.info("Azure OpenAI client initialized.")

logging.info("Loading Q&A data from CSV...")
try:
    qa_data = pd.read_csv("qa_data.csv", encoding="windows-1256")
    logging.info("CSV file loaded successfully!")
except Exception as e:
    logging.error("Failed to load CSV file: %s", e)
    exit()

logging.info("Normalizing column names...")
qa_data.rename(columns=lambda x: x.strip().lower(), inplace=True)
logging.info("Column names normalized.")

if "id" in qa_data.columns:
    qa_data["id"] = qa_data["id"].astype(str)
    logging.info("'id' column converted to string.")

logging.info("Verifying required columns...")
if "question" not in qa_data.columns or "answer" not in qa_data.columns:
    logging.error("CSV file must contain 'question' and 'answer' columns.")
    exit()
logging.info("Required columns are present.")

# EMBEDDING GENERATION
def get_embedding(text):
    logging.info("Generating embedding for text: %s", text)
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = response.data[0].embedding
        logging.info("Embedding generated for text: %s", text)
        return embedding
    except Exception as e:
        logging.error("Failed to generate embedding for text '%s': %s", text, e)
        return None

# Generate embeddings if not already present
if "embedding" not in qa_data.columns or qa_data["embedding"].isnull().all():
    logging.info("No embeddings found in CSV. Generating embeddings...")
    qa_data["embedding"] = qa_data["question"].apply(get_embedding)
    qa_data.to_csv("embedded_qa_data.csv", index=False)
    logging.info("Embeddings generated and saved to 'embedded_qa_data.csv'.")
else:
    logging.info("Embeddings column exists. Converting embeddings from CSV...")
    def convert_embedding(x):
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                logging.info("Embedding parsed successfully.")
                return parsed
            except Exception as e:
                logging.error("Failed to parse embedding: %s", e)
                return None
        return x
    qa_data["embedding"] = qa_data["embedding"].apply(convert_embedding)
    logging.info("Using existing embeddings from CSV.")

logging.info("Normalizing question text for matching...")
qa_data["question"] = qa_data["question"].str.strip().str.lower()
logging.info("Question text normalized.")

logging.info("Initializing Azure Cognitive Search client...")
search_client = SearchClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net/",
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)
logging.info("Azure Cognitive Search client initialized.")

logging.info("Converting Q&A data to documents for Azure Search...")
documents = qa_data.to_dict(orient="records")
logging.info("Documents converted to dictionary.")

try:
    upload_result = search_client.upload_documents(documents=documents)
    logging.info("Uploaded %s documents to Azure Search.", len(documents))
except Exception as e:
    logging.error("Failed to upload documents: %s", e)

logging.info("Initializing Redis client...")
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        ssl=True,
        decode_responses=True,
        password=REDIS_PASSWORD
    )
    redis_client.ping()
    logging.info("Successfully connected to Redis!")
except Exception as e:
    logging.error("Failed to connect to Redis: %s", e)

def check_redis_cache(query):
    logging.info("Checking Redis cache for query: %s", query)
    try:
        cached_answer = redis_client.get(query)
        if cached_answer:
            logging.info("Using cached answer for query: %s", query)
            return cached_answer
    except Exception as e:
        logging.error("Redis error: %s", e)
    logging.info("No cached answer found.")
    return None

def get_best_match(query):
    logging.info("Getting best match for query: %s", query)
    cached_response = check_redis_cache(query)
    if cached_response:
        return cached_response

    # --- Semantic Search ---
    logging.info("Performing Semantic Search...")
    try:
        semantic_results = search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="my-semantic-config-default",
            query_caption="extractive",
            select=["question", "answer"],
            top=3
        )
        logging.info("Semantic search executed.")
        semantic_answer = next(semantic_results, None)
        if semantic_answer:
            reranker_score = semantic_answer.get("@search.reranker_score", None)
            if reranker_score is not None and reranker_score >= SEMANTIC_THRESHOLD:
                answer = semantic_answer["answer"]
                logging.info("Found match using Semantic Search with score %s", reranker_score)
                redis_client.set(query, answer, ex=3600)
                return answer
            else:
                logging.info("Semantic search result score below threshold: %s", reranker_score)
        else:
            logging.info("No semantic search answers found.")
    except Exception as e:
        logging.error("Semantic search error: %s", e)

    # --- Vector Search ---
    logging.info("Performing Vector Search...")
    try:
        query_embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        ).data[0].embedding
        logging.info("Query embedding generated for vector search.")

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=50,
            fields="embedding"
        )
        logging.info("Constructed vector query.")

        vector_results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["question", "answer"],
            top=3
        )
        logging.info("Vector search executed.")
        best_vector = next(vector_results, None)
        if best_vector:
            score = best_vector.get("@search.score", 0)
            if score >= VECTOR_THRESHOLD:
                answer = best_vector["answer"]
                logging.info("Found match using Vector Search with score %s", score)
                redis_client.set(query, answer, ex=3600)
                return answer
            else:
                logging.info("Vector search result score below threshold: %s", score)
        else:
            logging.info("No vector search results found.")
    except Exception as e:
        logging.error("Vector search error: %s", e)

    logging.info("No match found using Semantic or Vector Search")
    return None

# GPT‑4o REALTIME FALLBACK (ASYNC)
async def get_realtime_response(user_query):
    logging.info("Attempting GPT‑4o realtime fallback for query: %s", user_query)
    try:
        async with RTLowLevelClient(
            url=RT_ENDPOINT,
            azure_deployment=RT_DEPLOYMENT,
            key_credential=AzureKeyCredential(RT_API_KEY)
        ) as client_rt:
            instructions = "أنت رجل عربي. انا لا اريد ايضا اي bold points  في الاجابة  و لا اريد عنواين مرقمة" + user_query
            logging.info("Sending realtime request with instructions.")
            await client_rt.send(
                ResponseCreateMessage(
                    response=ResponseCreateParams(
                        modalities={"text"},
                        instructions=instructions
                    )
                )
            )
            done = False
            response_text = ""
            while not done:
                message = await client_rt.recv()
                if message is None:
                    logging.error("No message received from the real-time service.")
                    break
                logging.info("Received message of type: %s", message.type)
                if message.type == "response.done":
                    done = True
                    logging.info("Realtime response completed.")
                elif message.type == "error":
                    done = True
                    logging.error("Error in realtime response: %s", message.error)
                elif message.type == "response.text.delta":
                    response_text += message.delta
                    logging.info("Received delta: %s", message.delta)
            return response_text
    except Exception as e:
        logging.error("Failed to get real-time response: %s", e)
        return "عذرًا، حدث خطأ أثناء محاولة الاتصال بخدمة الدعم الفوري. يرجى المحاولة مرة أخرى لاحقًا."

async def get_response(user_query):
    logging.info("Processing query: %s", user_query)
    response = get_best_match(user_query)
    if response:
        logging.info("Found response in cache or search: %s", response)
        return response

    logging.info("No match found, falling back to GPT‑4o realtime...")
    realtime_response = await get_realtime_response(user_query)
    if realtime_response:
        logging.info("GPT‑4o realtime response: %s", realtime_response)
        try:
            redis_client.set(user_query, realtime_response, ex=3600)
            logging.info("Response cached in Redis.")
        except Exception as e:
            logging.error("Failed to cache response in Redis: %s", e)
        return realtime_response
    else:
        logging.info("Realtime response empty.")
        return "عذرًا، لم أتمكن من العثور على إجابة. يرجى المحاولة مرة أخرى لاحقًا."

logging.info("Setting up Speech configuration...")
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "ar-EG"
speech_config.speech_synthesis_voice_name = "ar-EG-ShakirNeural"
logging.info("Speech configuration set.")

def recognize_speech():
    logging.info("Starting speech recognition...")
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    logging.info("Listening... (Speak in Egyptian Arabic)")
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        logging.info("Speech recognized: %s", result.text)
        return result.text
    else:
        logging.error("Speech not recognized: %s", result.reason)
        return ""

def speak_response(text):
    logging.info("Speaking response: %s", text)
    audio_output = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        logging.error("Speech synthesis canceled: Reason: %s, Error Details: %s", cancellation.reason, cancellation.error_details)
    else:
        logging.info("Speech synthesis completed.")

def clean_text(text):
    cleaned = text.strip(" .،!؛؟").lower()
    logging.info("Cleaning text. Before: '%s' After: '%s'", text, cleaned)
    return cleaned

def detect_critical_issue(text):
    logging.info("Detecting critical issue in text: %s", text)
    trigger_sentences = [
        "تم اكتشاف اختراق أمني كبير.",
        "تمكن قراصنة من الوصول إلى بيانات حساسة.",
        "هناك هجوم إلكتروني على النظام الخاص بنا.",
        "تم تسريب بيانات المستخدمين إلى الإنترنت.",
        "رصدنا محاولة تصيد إلكتروني ضد موظفينا.",
        "تم استغلال ثغرة أمنية في الشبكة.",
        "هناك محاولة وصول غير مصرح بها إلى الملفات السرية."
    ]
    trigger_embeddings = np.array([get_embedding(sent) for sent in trigger_sentences])
    text_embedding = np.array(get_embedding(text)).reshape(1, -1)
    similarities = cosine_similarity(text_embedding, trigger_embeddings)
    max_similarity = np.max(similarities)
    logging.info("Maximum similarity with trigger sentences: %s", max_similarity)
    if max_similarity > 0.9:
        logging.info("Critical issue detected. This issue should be passed to a human.")
        return True
    logging.info("No critical issue detected.")
    return False

async def voice_chat(turn_context: TurnContext, user_query: str):
    logging.info("Voice chat started with query: %s", user_query)
    if not user_query:
        logging.info("Empty user query received.")
        return "في انتظار اوامرك"
    if clean_text(user_query) in ["إنهاء", "خروج"]:
        logging.info("Goodbye command received.")
        return "مع السلامة"
    if detect_critical_issue(user_query):
        logging.info("Critical issue detected in voice chat.")
        return "هذه المشكلة تحتاج إلى تدخل بشري. سأقوم بالاتصال بخدمة العملاء لدعمك."
    response = await get_response(user_query)
    logging.info("Voice chat response: %s", response)
    activity: Activity = turn_context.activity
    bot_id = activity.recipient.id
    return Activity(
        type=ActivityTypes.message,
        from_property=ChannelAccount(id="lcw"),  # Bot as the sender
        text=response
    )

class MyBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        user_query = turn_context.activity.text
        logging.info("Received message: %s", user_query)
        response_text = await voice_chat(turn_context, user_query)
        logging.info("Sending response back to user.")
        logging.info("response: %s", response_text)
        await turn_context.send_activity(response_text)

    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        logging.info("New members added to the conversation.")
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                welcome_activity = Activity(               
                    type=ActivityTypes.message,
                    from_property=ChannelAccount(
                        id="8:bot:ms-poc-contact-center-voice-bot"
                    ),
                    text="مرحبًا! كيف يمكنني مساعدتك اليوم؟"
                )
                logging.info("welcome_activity: %s", welcome_activity)
                logging.info("Sending welcome message.")
                await turn_context.send_activity(welcome_activity)

logging.info("All modules and configurations are set. Bot is ready to run!")
