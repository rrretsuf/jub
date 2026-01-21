import asyncio
import base64
import os
import traceback
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

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


class EditRequest(BaseModel):
    original_image: str
    generated_image: str
    edit_comments: str
    color_name: str
    color_hex: str


def _add_watermark(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text = "ORIGINAL"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=min(img.width, img.height) // 8)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2
    draw.text((x, y), text, fill=(255, 255, 255, 180), font=font)
    watermarked = Image.alpha_composite(img, overlay)
    watermarked = watermarked.convert("RGB")
    buffer = BytesIO()
    watermarked.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


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


def _generate_edit_prompt_sync(
    api_key: str,
    original_watermarked_bytes: bytes,
    generated_bytes: bytes,
    edit_comments: str,
    color_name: str,
    color_hex: str,
) -> str:
    print("[Flash Edit] Generating edit prompt...")
    client = genai.Client(api_key=api_key)
    flash_system_prompt = _load_flash_system_prompt()
    template = _load_image_system_prompt_template()

    edit_input = f"""This is an EDIT request. The user already generated a colored wall image but is not satisfied and wants corrections.

TWO images are provided:
1. Image with "ORIGINAL" watermark = the original reference photo (unchanged room)
2. Other image = the AI-generated colored version that needs editing

Your task: Analyze BOTH images, understand what the user wants changed based on their comments, and write a prompt for the image-editing model. The image model will receive ONLY the generated image + your prompt, so your prompt must be complete and precise.

Use this TEMPLATE as the base structure for your edit prompt:
{template}

---

This is an edit message from the user. You must analyse the original image and the generated image, figure out what went wrong or what the user wants different, and then write a prompt for the image model based on the template and the user's edit needs so they can fix their image.

USER EDIT COMMENTS: {edit_comments}

COLOR CONTEXT: {color_name} ({color_hex})

---

Common Slovenian edit requests:
- "strop ne sme biti pobarvan" = ceiling should NOT be painted, restore to original
- "kamin mora ostati nespremenjen" = fireplace must stay as in original
- "okno ne sme biti pobarvano" = window frame should not be painted
- "pobarvaj tudi X" = also paint X in the same color

Generate a complete, precise prompt for the image model. Output ONLY the prompt text."""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=edit_input),
                types.Part.from_bytes(data=original_watermarked_bytes, mime_type="image/png"),
                types.Part.from_bytes(data=generated_bytes, mime_type="image/png"),
            ],
        ),
    ]
    config = types.GenerateContentConfig(
        system_instruction=flash_system_prompt,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    edit_prompt = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-flash-latest",
        contents=contents,
        config=config,
    ):
        if chunk.text:
            edit_prompt += chunk.text

    edit_prompt = edit_prompt.strip()
    print(f"[Flash Edit] Prompt ready ({len(edit_prompt)} chars)")
    return edit_prompt


def _generate_edit_sync(
    api_key: str,
    generated_bytes: bytes,
    edit_prompt: str,
    color_hex: str,
) -> dict:
    print("[Edit] Starting edit generation...")
    client = genai.Client(api_key=api_key)

    full_prompt = f"""{edit_prompt}

Color to use: {color_hex}
Apply the requested changes to this image."""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=full_prompt),
                types.Part.from_bytes(data=generated_bytes, mime_type="image/png"),
            ],
        )
    ]
    config = types.GenerateContentConfig(response_modalities=["IMAGE"])

    output_image_data = None
    output_mime = None

    for chunk in client.models.generate_content_stream(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=config,
    ):
        if chunk.candidates is None or chunk.candidates[0].content is None:
            continue
        if chunk.candidates[0].content.parts is None:
            continue
        for part in chunk.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                output_image_data = part.inline_data.data
                output_mime = part.inline_data.mime_type
                break
        if output_image_data:
            break

    if not output_image_data:
        return {"error": "No edited image returned"}

    print(f"[Edit] SUCCESS - image size: {len(output_image_data)} bytes")
    b64 = base64.b64encode(output_image_data).decode("utf-8")
    mime = output_mime or "image/png"
    return {"image": f"data:{mime};base64,{b64}"}


@app.post("/api/edit")
async def edit_image(req: EditRequest) -> JSONResponse:
    try:
        if not req.edit_comments.strip():
            return JSONResponse({"error": "Edit comments are required"}, status_code=400)

        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            return JSONResponse({"error": "GENAI_API_KEY not set"}, status_code=500)

        print("=" * 50)
        print(f"Edit request: {req.edit_comments[:100]}...")

        original_bytes, original_mime = _data_url_to_bytes(req.original_image)
        generated_bytes, _ = _data_url_to_bytes(req.generated_image)

        watermarked_bytes, _ = _add_watermark(original_bytes, original_mime)

        edit_prompt = await asyncio.to_thread(
            _generate_edit_prompt_sync,
            api_key,
            watermarked_bytes,
            generated_bytes,
            req.edit_comments.strip(),
            req.color_name,
            req.color_hex,
        )

        result = await asyncio.to_thread(
            _generate_edit_sync,
            api_key,
            generated_bytes,
            edit_prompt,
            req.color_hex,
        )

        print("=" * 50)
        return JSONResponse({"result": result})

    except Exception as e:
        print("Error processing edit request:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)