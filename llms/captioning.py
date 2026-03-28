import base64
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

def generate_image_caption(image_path: str) -> str:
    """Generate a detailed caption for the image at image_path."""
    # Initialize client here to prevent the whole server from crashing on startup if the key is missing
    client = Groq(api_key=os.getenv("GROQAPI"))
    
    with open(image_path, "rb") as f:  # Absolute path recommended
        img_data = base64.b64encode(f.read()).decode("utf-8")

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a detailed caption for this technical image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}",
                        },
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content
