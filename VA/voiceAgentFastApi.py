from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import uvicorn
import whisper
from transformers import pipeline 
import pyttsx3

app = FastAPI(title="Voice Agent API", description="API for voice agent functionalities")

# Global variables for lazy loading
asr_model = None
llm = None
tts_engine = None
conversation_history = []

def get_asr_model():
    global asr_model
    if asr_model is None:
        print("Loading Whisper model...")
        asr_model = whisper.load_model("small")
        print("Whisper model loaded successfully")
    return asr_model

def get_llm():
    global llm
    if llm is None:
        print("Loading language model...")
        llm = pipeline("text-generation", "mistralai/Mistral-7B-Instruct-v0.1")
        print("Language model loaded successfully")
    return llm

def get_tts_engine():
    global tts_engine
    if tts_engine is None:
        print("Initializing TTS engine...")
        tts_engine = pyttsx3.init()
        print("TTS engine initialized successfully")
    return tts_engine

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Voice Agent FastAPI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/chat/": "POST endpoint to upload audio files for transcription (mock response)",
            "/health": "GET endpoint for health check",
            "/docs": "Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Voice Agent API is running"}

def transcribe_audio(audio_bytes):
    asr_model = get_asr_model()
    with open("temporary.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temporary.wav")
    return result["text"]

def generate_response(user_text):
    llm = get_llm()
    conversation_history.append({"role": "user", "text": user_text})
    # Construct prompt from history
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    outputs = llm(prompt, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response

def synthesize_speech(text, filename="response.wav"):
    tts_engine = get_tts_engine()
    tts_engine.save_to_file(text, filename)
    tts_engine.runAndWait()
    return filename

@app.post("/chat/")
async def chat_endpoint(request: Request):
    """
    Chat endpoint that accepts audio files
    below curl line, @ symbol indicates file content
    # curl -X POST -H "Content-Type: audio/wav" --data-binary "@C:\\works\\ai\\VA\\temp1.wav" http://localhost:8002/chat/
    """
    try:
        # Get the raw body content
        body = await request.body()
        
        if not body:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # For now, return a mock response with the body size
        file_size = len(body)
        
        # Transcribe audio and generate response
        transcribed_text = transcribe_audio(body)
        bot_text = generate_response(transcribed_text)
        audio_path=synthesize_speech(bot_text)        
        return FileResponse(audio_path, media_type="audio/wav", filename="response.wav")
        # return {
        #     "transcribed_text": f"Mock transcription: {transcribed_text} - Received audio file ({file_size} bytes)",
        #     "bot_response": bot_text,
        #     "status": "success"
        # }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
if __name__ == "__main__":
    print("Starting Voice Agent API...")
    print("API will be available at: http://localhost:8002")
    print("Interactive docs at: http://localhost:8002/docs")
    uvicorn.run(app, host="localhost", port=8002)
