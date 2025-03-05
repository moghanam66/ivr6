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
nest_asyncio.apply()

# Azure Cognitive Search imports
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# Bot Framework dependencies
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount , Activity, ActivityTypes
# RT client for GPT‑4o realtime fallback (make sure the rtclient package is installed)
from rtclient import RTLowLevelClient, ResponseCreateMessage, ResponseCreateParams

# ------------------------------------------------------------------
# Configuration for Azure OpenAI, GPT‑4o realtime, Azure Search, Redis, Speech
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# Initialize clients and load Q&A data
# ------------------------------------------------------------------

# Initialize the Azure OpenAI client (for embeddings)
client = openai.AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)

# Load Q&A data from CSV (ensure the file exists and contains 'question' and 'answer' columns)
try:
    qa_data = pd.read_csv("qa_data.csv", encoding="windows-1256")
    print("✅ CSV file loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load CSV file: {e}")
    exit()

# Normalize column names
qa_data.rename(columns=lambda x: x.strip().lower(), inplace=True)

# Convert 'id' column to string if it exists
if "id" in qa_data.columns:
    qa_data["id"] = qa_data["id"].astype(str)

# Verify required columns exist
if "question" not in qa_data.columns or "answer" not in qa_data.columns:
    print("❌ CSV file must contain 'question' and 'answer' columns.")
    exit()

# EMBEDDING GENERATION
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = response.data[0].embedding
        print(f"✅ Embedding generated for text: {text}")
        return embedding
    except Exception as e:
        print(f"❌ Failed to generate embedding for text '{text}': {e}")
        return None

# Generate embeddings if not already present
if "embedding" not in qa_data.columns or qa_data["embedding"].isnull().all():
    qa_data["embedding"] = qa_data["question"].apply(get_embedding)
    qa_data.to_csv("embedded_qa_data.csv", index=False)
    print("✅ Embeddings generated and saved to 'embedded_qa_data.csv'.")
else:
    def convert_embedding(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception as e:
                print("❌ Failed to parse embedding:", e)
                return None
        return x
    qa_data["embedding"] = qa_data["embedding"].apply(convert_embedding)
    print("✅ Using existing embeddings from CSV.")

# Normalize question text for matching
qa_data["question"] = qa_data["question"].str.strip().str.lower()

# Initialize Azure Cognitive Search client
search_client = SearchClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net/",
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Upload documents to Azure Cognitive Search
documents = qa_data.to_dict(orient="records")
try:
    upload_result = search_client.upload_documents(documents=documents)
    print(f"✅ Uploaded {len(documents)} documents to Azure Search.")
except Exception as e:
    print(f"❌ Failed to upload documents: {e}")

# Initialize Redis client
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
    print("✅ Successfully connected to Redis!")
except Exception as e:
    print(f"❌ Failed to connect to Redis: {e}")



def check_redis_cache(query):
    try:
        cached_answer = redis_client.get(query)
        if cached_answer:
            print(f"✅ Using cached answer for query: {query}")
            return cached_answer
    except Exception as e:
        print(f"❌ Redis error: {e}")
    return None

def get_best_match(query):
    cached_response = check_redis_cache(query)
    if cached_response:
        return cached_response

    # --- Semantic Search ---
    try:
        semantic_results = search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="my-semantic-config-default",
            query_caption="extractive",
            select=["question", "answer"],
            top=3
        )
        semantic_answer = next(semantic_results, None)
        if semantic_answer:
            reranker_score = semantic_answer.get("@search.reranker_score", None)
            if reranker_score is not None and reranker_score >= SEMANTIC_THRESHOLD:
                answer = semantic_answer["answer"]
                print("✅ Found match using Semantic Search with score", reranker_score)
                redis_client.set(query, answer, ex=3600)
                return answer
            else:
                print("❌ Semantic search result score below threshold:", reranker_score)
        else:
            print("❌ No semantic search answers found.")
    except Exception as e:
        print(f"❌ Semantic search error: {e}")

    # --- Vector Search ---
    try:
        query_embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        ).data[0].embedding

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=50,
            fields="embedding"
        )
        vector_results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["question", "answer"],
            top=3
        )
        best_vector = next(vector_results, None)
        if best_vector:
            score = best_vector.get("@search.score", 0)
            if score >= VECTOR_THRESHOLD:
                answer = best_vector["answer"]
                print("✅ Found match using Vector Search with score", score)
                redis_client.set(query, answer, ex=3600)
                return answer
            else:
                print("❌ Vector search result score below threshold:", score)
        else:
            print("❌ No vector search results found.")
    except Exception as e:
        print(f"❌ Vector search error: {e}")

    print("❌ No match found using Semantic or Vector Search")
    return None

# GPT‑4o REALTIME FALLBACK (ASYNC)
async def get_realtime_response(user_query):
    try:
        async with RTLowLevelClient(
            url=RT_ENDPOINT,
            azure_deployment=RT_DEPLOYMENT,
            key_credential=AzureKeyCredential(RT_API_KEY)
        ) as client_rt:
            # Prepend instruction for an Egyptian persona
            instructions = "أنت رجل عربي. انا لا اريد ايضا اي bold points  في الاجابة  و لا اريد عنواين مرقمة" + user_query
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
                    print("❌ No message received from the real-time service.")
                    break
                if message.type == "response.done":
                    done = True
                elif message.type == "error":
                    done = True
                    print(f"Error: {message.error}")
                elif message.type == "response.text.delta":
                    response_text += message.delta
            return response_text
    except Exception as e:
        print(f"❌ Failed to get real-time response: {e}")
        return "عذرًا، حدث خطأ أثناء محاولة الاتصال بخدمة الدعم الفوري. يرجى المحاولة مرة أخرى لاحقًا."

async def get_response(user_query):
    print(f"🔍 Processing query: {user_query}")
    response = get_best_match(user_query)
    if response:
        print(f"✅ Found response in cache or search: {response}")
        return response

    print("🔍 No match found, falling back to GPT‑4o realtime...")
    realtime_response = await get_realtime_response(user_query)
    if realtime_response:
        print(f"✅ GPT‑4o realtime response: {realtime_response}")
        try:
            redis_client.set(user_query, realtime_response, ex=3600)
            print("✅ Response cached in Redis.")
        except Exception as e:
            print(f"❌ Failed to cache response in Redis: {e}")
        return realtime_response
    else:
        return "عذرًا، لم أتمكن من العثور على إجابة. يرجى المحاولة مرة أخرى لاحقًا."



speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "ar-EG"
speech_config.speech_synthesis_voice_name = "ar-EG-ShakirNeural"

def recognize_speech():
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    print("Listening... (Speak in Egyptian Arabic)")
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"You said: {result.text}")
        return result.text
    else:
        print(f"Speech not recognized: {result.reason}")
        return ""

def speak_response(text):
    audio_output = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print("Speech synthesis canceled:")
        print("  Reason: {}".format(cancellation.reason))
        print("  Error Details: {}".format(cancellation.error_details))


def clean_text(text):
    return text.strip(" .،!؛؟").lower()

def detect_critical_issue(text):
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
    if max_similarity > 0.9:
        print("This issue should be passed to a human.")
        return True
    return False


async def voice_chat(turn_context: TurnContext, user_query: str):
    if not user_query:
        return "في انتظار اوامرك"
    if clean_text(user_query) in ["إنهاء", "خروج"]:
        print("👋 Goodbye!")
        return "مع السلامة"
    if detect_critical_issue(user_query):
        return "هذه المشكلة تحتاج إلى تدخل بشري. سأقوم بالاتصال بخدمة العملاء لدعمك."
    response = await get_response(user_query)
    activity: Activity = turn_context.activity
    bot_id = activity.recipient.id
    return Activity(
    type=ActivityTypes.message,
    from_property=ChannelAccount(id=bot_id),  # Bot as the sender
    text=response)
    # return response



class MyBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        user_query = turn_context.activity.text
        print(f"Received message: {user_query}")
        response_text =  await voice_chat(turn_context, user_query)
        await turn_context.send_activity(response_text)

    async def on_members_added_activity(
        self, members_added: ChannelAccount, turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                welcome_message = "مرحبًا! كيف يمكنني مساعدتك اليوم؟"
                await turn_context.send_activity(welcome_message)
