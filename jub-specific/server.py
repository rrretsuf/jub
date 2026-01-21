import asyncio
import base64
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

SHADES = {
    "beauty-15": {
        "name": "Beauty 15",
        "rgb": [230, 169, 115],
        "hex": "#E6A973",
        "url": "https://www.jub.si/barve/beauty-15/",
    },
    "joy-15": {
        "name": "Joy 15",
        "rgb": [255, 229, 102],
        "hex": "#FFE566",
        "url": "https://www.jub.si/barve/joy-15/",
    },
    "family-40": {
        "name": "Family 40",
        "rgb": [143, 112, 101],
        "hex": "#8F7065",
        "url": "https://www.jub.si/barve/family-40/",
    },
    "vitality-45": {
        "name": "Vitality 45",
        "rgb": [97, 184, 162],
        "hex": "#61B8A2",
        "url": "https://www.jub.si/barve/vitality-45/",
    },
    "wisdom-40": {
        "name": "Wisdom 40",
        "rgb": [181, 67, 117],
        "hex": "#B54375",
        "url": "https://www.jub.si/barve/wisdom-40/",
    },
}


class ColorizeRequest(BaseModel):
    original_image: str
    wall_description: str
    shade_id: str = "beauty-15"


def _load_image_system_prompt_template() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "image_system_prompt_template.md"
    return prompt_path.read_text(encoding="utf-8")


def _load_flash_system_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "flash_system_prompt.md"
    return prompt_path.read_text(encoding="utf-8")


def _data_url_to_bytes(data_url: str) -> tuple[bytes, str]:
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    header, b64_data = data_url.split(",", 1)
    raw = base64.b64decode(b64_data)
    mime_type = "image/png"
    if header.startswith("data:"):
        mime_type = header.split(";", 1)[0].split(":", 1)[1]
    return raw, mime_type


def _build_prompt(dynamic_system_prompt: str, wall_description: str, color_def: dict) -> str:
    rgb = color_def["rgb"]
    hex_value = color_def["hex"]
    return f"""{dynamic_system_prompt}

User instructions: {wall_description}
Color to use: RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) / HEX: {hex_value}"""


def _generate_dynamic_system_prompt_sync(
    api_key: str,
    original_bytes: bytes,
    original_mime: str,
    wall_description: str,
) -> str:
    """Generate a scene-specific system prompt using Gemini Flash."""
    template = _load_image_system_prompt_template()
    client = genai.Client(api_key=api_key)

    print("[Flash] Generating dynamic system prompt...")
    flash_system_prompt = _load_flash_system_prompt()

    flash_input = f"""You are generating a system prompt for an image-editing model.
Use the TEMPLATE below as the base, adapt it to this specific image and the user's instructions.
Keep the same rules and intent, but add scene-specific cautions if needed.
Explicitly handle reflections: if walls appear in mirrors/glass, the wall color must be updated in the reflection too.
Output ONLY the final system prompt. No markdown, no explanations.

TEMPLATE:
{template}

USER INSTRUCTIONS:
{wall_description}
"""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=flash_input),
                types.Part.from_bytes(data=original_bytes, mime_type=original_mime),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        system_instruction=flash_system_prompt,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
    )

    generated_prompt = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-flash-latest",
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            generated_prompt += chunk.text

    generated_prompt = generated_prompt.strip()
    if not generated_prompt:
        raise ValueError("Flash prompt generation returned empty text")

    print(f"[Flash] Dynamic system prompt ready ({len(generated_prompt)} chars)")
    return generated_prompt


def _generate_single_color_sync(
    api_key: str,
    original_bytes: bytes,
    original_mime: str,
    wall_description: str,
    color_def: dict,
    dynamic_system_prompt: str,
) -> dict:
    """Generate image for a single color (synchronous)."""
    color_name = color_def["name"]
    print(f"[{color_name}] Starting generation...")
    
    try:
        client = genai.Client(api_key=api_key)
        prompt = _build_prompt(dynamic_system_prompt, wall_description, color_def)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=original_bytes, mime_type=original_mime),
                ],
            )
        ]
        config = types.GenerateContentConfig(response_modalities=["IMAGE"])

        output_image_data = None
        output_mime = None
        text_response = ""

        for chunk in client.models.generate_content_stream(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=config,
        ):
            if chunk.candidates is None:
                print(f"[{color_name}] Chunk has no candidates")
                continue
            if chunk.candidates[0].content is None:
                print(f"[{color_name}] Candidate has no content")
                continue
            if chunk.candidates[0].content.parts is None:
                print(f"[{color_name}] Content has no parts")
                continue
                
            for part in chunk.candidates[0].content.parts:
                if part.text:
                    text_response += part.text
                if part.inline_data and part.inline_data.data:
                    output_image_data = part.inline_data.data
                    output_mime = part.inline_data.mime_type
                    print(f"[{color_name}] Got image data, mime: {output_mime}")
                    break
            if output_image_data:
                break

        if text_response:
            print(f"[{color_name}] Text response: {text_response[:200]}...")

        if not output_image_data:
            print(f"[{color_name}] NO IMAGE RETURNED")
            error_msg = "No image returned"
            if text_response:
                error_msg = f"Model returned text: {text_response[:100]}"
            return {"name": color_name, "hex": color_def["hex"], "error": error_msg}

        print(f"[{color_name}] SUCCESS - image size: {len(output_image_data)} bytes")
        b64 = base64.b64encode(output_image_data).decode("utf-8")
        mime = output_mime or "image/png"
        return {
            "name": color_name,
            "hex": color_def["hex"],
            "image": f"data:{mime};base64,{b64}",
        }

    except Exception as e:
        print(f"[{color_name}] ERROR: {e}")
        traceback.print_exc()
        return {"name": color_name, "hex": color_def["hex"], "error": str(e)}


@app.get("/")
def index() -> HTMLResponse:
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("index.html not found", status_code=404)


@app.get("/api/odtenki")
def get_shades() -> JSONResponse:
    """Return available shades."""
    return JSONResponse({
        "shades": [
            {
                "id": key,
                "name": value["name"],
                "hex": value["hex"],
                "rgb": value["rgb"],
                "url": value["url"],
            }
            for key, value in SHADES.items()
        ],
    })


@app.post("/api/colorize")
async def colorize(req: ColorizeRequest) -> JSONResponse:
    """Process wall coloring for selected shade."""
    try:
        shade_id = req.shade_id.lower()
        if shade_id not in SHADES:
            return JSONResponse({"error": f"Unknown shade: {shade_id}"}, status_code=400)

        color_def = SHADES[shade_id]
        print("=" * 50)
        print(f"Received test request for shade {color_def['name']}")

        if not req.wall_description.strip():
            return JSONResponse({"error": "Wall description is required"}, status_code=400)

        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            return JSONResponse({"error": "GENAI_API_KEY not set"}, status_code=500)

        original_bytes, original_mime = _data_url_to_bytes(req.original_image)
        wall_desc = req.wall_description.strip()
        dynamic_system_prompt = _generate_dynamic_system_prompt_sync(
            api_key=api_key,
            original_bytes=original_bytes,
            original_mime=original_mime,
            wall_description=wall_desc,
        )

        result = await asyncio.to_thread(
            _generate_single_color_sync,
            api_key,
            original_bytes,
            original_mime,
            wall_desc,
            color_def,
            dynamic_system_prompt,
        )

        print("=" * 50)
        print("Generation complete")
        return JSONResponse({"result": result})

    except Exception as e:
        print("Error processing test request:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)