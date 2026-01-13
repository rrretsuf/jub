import base64
import json
import mimetypes
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# Serve color definitions for frontend sync
app.mount("/color-defs", StaticFiles(directory="color-defs"), name="color-defs")


class ColorizeRequest(BaseModel):
    original_image: str  # data URL of clean original image
    color: int  # 1-5 (color selection)
    wall_description: str  # text description of wall(s) to paint


def _load_color_defs() -> dict:
    defs_path = Path(__file__).parent / "color-defs" / "colors.json"
    return json.loads(defs_path.read_text(encoding="utf-8"))


def _data_url_to_bytes(data_url: str) -> tuple[bytes, str]:
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    header, b64_data = data_url.split(",", 1)
    raw = base64.b64decode(b64_data)
    mime_type = "image/png"
    if header.startswith("data:"):
        mime_type = header.split(";", 1)[0].split(":", 1)[1]
    return raw, mime_type


def _build_prompt(wall_description: str, color_def: dict) -> str:
    rgb = color_def["rgb"]
    hex_value = color_def["hex"]
    return f"""pobarvaj samo {wall_description} z attachano barvo sliko ki je attached
tvoj cilj je da obarvaš stene (ki jih user hoče da so pobarvane), točno tako kot ta barva:
RGB
{rgb[0]} {rgb[1]} {rgb[2]}
HEX
{hex_value}
ne spreminjaj nič drugega
ne spremeni niti pixla kar koli kar ni stena!
fokus samo na barvo stene!
okej?
pazi na lighting in sence (da ne bo preveč temno in svetlo / fake)
bodi pozoren na predmeta in da barvaš samo steno in nič drugega kot steno!!!!! SAMO STENO BARVAŠ!
go!"""


@app.get("/")
def index() -> HTMLResponse:
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("index.html not found", status_code=404)


@app.post("/api/colorize")
async def colorize(req: ColorizeRequest) -> JSONResponse:
    """Process wall coloring request with Google Gemini image model."""
    try:
        print(f"Received request for color {req.color}")

        # Validate color selection
        if req.color < 1 or req.color > 5:
            return JSONResponse(
                {"error": "Color must be between 1 and 5"},
                status_code=400
            )

        if not req.wall_description.strip():
            return JSONResponse(
                {"error": "Wall description is required"},
                status_code=400
            )

        # Check API key
        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            return JSONResponse(
                {"error": "GENAI_API_KEY environment variable not set"},
                status_code=500
            )

        color_defs = _load_color_defs()
        color_def = color_defs.get(str(req.color))
        if not color_def:
            return JSONResponse(
                {"error": "Color definition not found"},
                status_code=400
            )

        original_bytes, original_mime = _data_url_to_bytes(req.original_image)
        prompt = _build_prompt(req.wall_description.strip(), color_def)

        client = genai.Client(api_key=api_key)
        model = "gemini-3-pro-image-preview"

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=original_bytes, mime_type=original_mime),
                ],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(image_size="1K"),
        )

        output_image_data = None
        output_mime = None

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            for part in chunk.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    output_image_data = part.inline_data.data
                    output_mime = part.inline_data.mime_type
                    break
            if output_image_data:
                break

        if not output_image_data:
            return JSONResponse(
                {"error": "Gemini did not return an image. Check server logs."},
                status_code=500
            )

        file_extension = mimetypes.guess_extension(output_mime or "image/png") or ".png"
        b64 = base64.b64encode(output_image_data).decode("utf-8")
        mime_type = output_mime or "image/png"
        return JSONResponse({"image": f"data:{mime_type};base64,{b64}", "ext": file_extension})

    except Exception as e:
        print("Error processing request:")
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
