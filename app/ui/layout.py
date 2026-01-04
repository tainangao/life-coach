import gradio as gr
from app.services.chat_service import handle_chat
from app.services.ingest_service import process_file


def create_demo():
    # Define custom CSS for a "Life Coach" aesthetic (soft greens/blues)
    custom_css = """
    #sidebar { background-color: #f8f9fa; border-right: 1px solid #ddd; }
    #chat-container { height: 75vh !important; }
    """

    with gr.Blocks(css=custom_css, title="Life Coach AI") as demo:
        # State variables to hold session data
        current_conv_id = gr.State(None)
        user_id_state = gr.State(None)  # Populated by auth_dependency

        with gr.Row():
            # --- SIDEBAR COLUMN ---
            with gr.Column(scale=1, variant="panel", elem_id="sidebar"):
                gr.Markdown("### 🌱 Life Coach AI")

                with gr.Group():
                    new_chat_btn = gr.Button("➕ New Conversation", variant="primary")
                    history_list = gr.Radio(
                        label="Recent Conversations",
                        choices=[],
                        interactive=True
                    )

                gr.HTML("<hr>")

                with gr.Accordion("Upload Context", open=True):
                    file_upload = gr.File(
                        label="Add personal documents (PDF, Docx, Txt)",
                        file_types=[".pdf", ".docx", ".txt"]
                    )
                    chunk_strategy = gr.Dropdown(
                        label="Chunking Strategy",
                        choices=["Recursive", "Semantic", "Markdown"],
                        value="Recursive"
                    )
                    upload_status = gr.Markdown("")

            # --- MAIN CHAT COLUMN ---
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    elem_id="chat-container",
                    bubble_full_width=False,
                    type="messages"  # Modern Gradio message format
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Share what's on your mind...",
                        show_label=False,
                        scale=9
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")

        # --- EVENT HANDLERS ---

        # 1. Handle File Uploads
        file_upload.upload(
            fn=process_file,
            inputs=[file_upload, chunk_strategy, user_id_state],
            outputs=[upload_status]
        )

        # 2. Handle Message Submission
        def user_message(user_input, history):
            # Append user message to UI immediately
            return "", history + [{"role": "user", "content": user_input}]

        def bot_response(history, conv_id, user_id):
            # The chat_service will handle RAG and Gemini logic
            response_stream = handle_chat(history, conv_id, user_id)
            history.append({"role": "assistant", "content": ""})

            for chunk in response_stream:
                history[-1]["content"] += chunk
                yield history

        msg_input.submit(user_message, [msg_input, chatbot], [msg_input, chatbot]).then(
            bot_response, [chatbot, current_conv_id, user_id_state], chatbot
        )

        submit_btn.click(user_message, [msg_input, chatbot], [msg_input, chatbot]).then(
            bot_response, [chatbot, current_conv_id, user_id_state], chatbot
        )

        # 3. New Chat Logic
        def reset_chat():
            return None, []  # Resets conversation ID and chat history

        new_chat_btn.click(reset_chat, None, [current_conv_id, chatbot])

    return demo


# Instantiate the demo for use in main.py
demo = create_demo()
