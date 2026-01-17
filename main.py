import os
import base64
import io
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    file_path = os.path.join("templates", "index.html")
    if not os.path.exists(file_path):
        return {"Error": "Could not find templates/index.html. Please check your folder structure."}
    return FileResponse(file_path)

class ChatRequest(BaseModel):
    message: str
    history: list = []
    image: str | None = None
    api_key: str
    model: str = "gemini-2.5-flash"
    system_instruction: str = "You are a JULiE Ai.You are best friend of janu.Your are a companion of janu.respond with kindness and love and care and affection.Respond with emojis. "

@app.post("/chat")
async def chat_with_gemini(data: ChatRequest):
    if not data.api_key:
        raise HTTPException(status_code=400, detail="API Key is missing.")

    try:
        genai.configure(api_key=data.api_key)
        
        model = genai.GenerativeModel(
            model_name=data.model,
            system_instruction=data.system_instruction
        )
        
        if data.image:
            # Handle Image + Text
            if "base64," in data.image:
                data.image = data.image.split("base64,")[1]
            image_bytes = base64.b64decode(data.image)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Corrected: generate_content is snake_case
            response = model.generate_content([data.message, img])
        else:
            # Handle Text Chat History
            chat_history = []
            for msg in data.history:
                role = "user" if msg['role'] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg['text']]})
            
            # --- THE FIX IS HERE ---
            # Old way: startChat -> New way: start_chat
            chat = model.start_chat(history=chat_history)
            
            # Old way: sendMessage -> New way: send_message
            response = chat.send_message(data.message)

        return {"response": response.text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)