import asyncio
import base64
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

COLOR_GROUPS = {
    "beauty": [
        {"name": "Beauty 05", "rgb": [179, 116, 51], "hex": "#B37433"},
        {"name": "Beauty 10", "rgb": [208, 145, 86], "hex": "#D09156"},
        {"name": "Beauty 15", "rgb": [230, 169, 115], "hex": "#E6A973"},
        {"name": "Beauty 20", "rgb": [243, 190, 147], "hex": "#F3BE93"},
        {"name": "Beauty 25", "rgb": [250, 207, 174], "hex": "#FACFAE"},
        {"name": "Beauty 30", "rgb": [254, 222, 197], "hex": "#FEDEC5"},
        {"name": "Beauty 31", "rgb": [255, 233, 215], "hex": "#FFE9D7"},
    ],
    "faith": [
        {"name": "Faith 05", "rgb": [112, 105, 96], "hex": "#706960"},
        {"name": "Faith 10", "rgb": [139, 138, 130], "hex": "#8B8A82"},
        {"name": "Faith 15", "rgb": [156, 155, 148], "hex": "#9C9B94"},
        {"name": "Faith 20", "rgb": [181, 179, 175], "hex": "#B5B3AF"},
        {"name": "Faith 25", "rgb": [200, 203, 200], "hex": "#C8CBC8"},
        {"name": "Faith 30", "rgb": [223, 224, 221], "hex": "#DFE0DD"},
        {"name": "Faith 31", "rgb": [236, 235, 233], "hex": "#ECEBE9"},
    ],
    "family": [
        {"name": "Family 95", "rgb": [104, 57, 56], "hex": "#683938"},
        {"name": "Family 100", "rgb": [133, 103, 108], "hex": "#85676C"},
        {"name": "Family 105", "rgb": [146, 121, 127], "hex": "#92797F"},
        {"name": "Family 110", "rgb": [174, 154, 161], "hex": "#AE9AA1"},
        {"name": "Family 115", "rgb": [191, 174, 180], "hex": "#BFAEB4"},
        {"name": "Family 120", "rgb": [217, 203, 209], "hex": "#D9CBD1"},
    ],
    "passion": [
        {"name": "Passion 35", "rgb": [239, 52, 36], "hex": "#EF3424"},
        {"name": "Passion 40", "rgb": [243, 74, 60], "hex": "#F34A3C"},
        {"name": "Passion 45", "rgb": [253, 109, 97], "hex": "#FD6D61"},
        {"name": "Passion 50", "rgb": [255, 139, 132], "hex": "#FF8B84"},
        {"name": "Passion 55", "rgb": [255, 160, 153], "hex": "#FFA099"},
        {"name": "Passion 60", "rgb": [255, 173, 167], "hex": "#FFADA7"},
        {"name": "Passion 61", "rgb": [249, 199, 192], "hex": "#F9C7C0"},
    ],
    "vitality": [
        {"name": "Vitality 05", "rgb": [0, 136, 121], "hex": "#008879"},
        {"name": "Vitality 10", "rgb": [0, 138, 133], "hex": "#008A85"},
        {"name": "Vitality 15", "rgb": [59, 168, 165], "hex": "#3BA8A5"},
        {"name": "Vitality 20", "rgb": [110, 191, 190], "hex": "#6EBFBE"},
        {"name": "Vitality 25", "rgb": [154, 213, 213], "hex": "#9AD5D5"},
        {"name": "Vitality 30", "rgb": [184, 230, 224], "hex": "#B8E6E0"},
        {"name": "Vitality 31", "rgb": [210, 238, 235], "hex": "#D2EEEB"},
    ],
}


class TestRequest(BaseModel):
    original_image: str
    wall_description: str
    color_group: str = "beauty"


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
            model="gemini-2.5-flash-image",
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


@app.get("/api/color-groups")
def get_color_groups() -> JSONResponse:
    """Return available color groups."""
    return JSONResponse({
        "groups": list(COLOR_GROUPS.keys()),
        "colors": COLOR_GROUPS,
    })


@app.post("/api/colorize-test")
async def colorize_test(req: TestRequest) -> JSONResponse:
    """Process wall coloring for selected color group."""
    try:
        color_group = req.color_group.lower()
        if color_group not in COLOR_GROUPS:
            return JSONResponse({"error": f"Unknown color group: {color_group}"}, status_code=400)

        colors = COLOR_GROUPS[color_group]
        print("=" * 50)
        print(f"Received test request for {len(colors)} colors ({color_group.upper()})")

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

        loop = asyncio.get_event_loop()
        
        # Limit to 3 parallel to avoid rate limits
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    _generate_single_color_sync,
                    api_key,
                    original_bytes,
                    original_mime,
                    wall_desc,
                    color,
                    dynamic_system_prompt,
                )
                for color in colors
            ]
            results = await asyncio.gather(*tasks)

        print("=" * 50)
        print(f"All {len(colors)} generations complete")
        return JSONResponse({"results": list(results)})

    except Exception as e:
        print("Error processing test request:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
