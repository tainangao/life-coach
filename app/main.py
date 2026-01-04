import gradio as gr
from app.ui.layout import demo  # Your Gradio Blocks
from fastapi import FastAPI, Request, HTTPException
from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth

app = FastAPI()

# 1. Attach HF OAuth endpoints (/oauth/huggingface/login, etc.)
attach_huggingface_oauth(app)


@app.get("/user/me")
def get_profile(request: Request):
    """Helper to check if user is logged in and return profile."""
    oauth_info = parse_huggingface_oauth(request)
    if not oauth_info:
        raise HTTPException(status_code=401, detail="Not logged in")
    return oauth_info.user_info


# 2. Mount Gradio to FastAPI
# We pass 'auth_dependency' to ensure Gradio knows who the user is.
def get_user_id(request: Request):
    oauth_info = parse_huggingface_oauth(request)
    return oauth_info.user_info.sub if oauth_info else None


app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    auth_dependency=get_user_id
)
