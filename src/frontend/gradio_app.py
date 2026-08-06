import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any

from src.core.rag_pipeline import RAGPipeline
from src.core.config import get_config
from src.core.session_manager import load_sessions_from_disk, save_sessions_to_disk

# Singleton pipeline
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

# Load sessions from disk
sessions = load_sessions_from_disk()
current_session_id = list(sessions.keys())[0] if sessions else "session_1"

# CSS design system (mirrors code.html / DESIGN.md)
MOCKUP_ZIP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

/* Reset & Override Dark mode for light background #f9f9ff */
html, body, .gradio-container, .dark, .dark .gradio-container {
    background-color: #f9f9ff !important;
    background: #f9f9ff !important;
    color: #161c27 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}

.gradio-container {
    padding: 0 !important;
    max-width: 100vw !important;
}

/* Header Top Bar sticky top */
#header, .dark #header {
    background-color: #f9f9ff !important;
    background: #f9f9ff !important;
    padding: 14px 28px !important;
    border-bottom: 1px solid rgba(108, 122, 118, 0.15) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}

#header_logo, .dark #header_logo {
    color: #006b5c !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

/* Sidebar Left (w-280px, bg #f1f3ff) */
#sidebar, .dark #sidebar, .dark .block {
    background-color: #f1f3ff !important;
    background: #f1f3ff !important;
    padding: 20px 16px !important;
    border-right: 1px solid rgba(108, 122, 118, 0.2) !important;
}

#brand_title, .dark #brand_title {
    color: #006b5c !important;
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    margin: 0 !important;
}

/* New Session Button (#00bfa5 rounded-xl font-bold) */
#new_session_btn, .dark #new_session_btn {
    background-color: #00bfa5 !important;
    background: #00bfa5 !important;
    color: #00473c !important;
    border: none !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(0, 191, 165, 0.25) !important;
    transition: all 0.2s ease !important;
}

#new_session_btn:hover {
    background-color: #00a896 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px rgba(0, 191, 165, 0.4) !important;
    transform: translateY(-1px);
}

/* Chat Header Sticky */
#chat_header, .dark #chat_header {
    background-color: rgba(249, 249, 255, 0.9) !important;
    padding: 16px 28px !important;
    border-bottom: 1px solid rgba(108, 122, 118, 0.15) !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}

#session_title, .dark #session_title {
    color: #00473c !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

#doc_info_chip, .dark #doc_info_chip {
    background-color: #e3e8f9 !important;
    color: #3c4a46 !important;
    padding: 6px 14px !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* Chat Canvas max-w-800px */
#chatbot, .dark #chatbot, .dark .chatbot {
    background-color: #f9f9ff !important;
    background: #f9f9ff !important;
    border: none !important;
    max-width: 800px !important;
    margin: 0 auto !important;
}

/* Suggestion Chips (bg #76f4e0 rounded-full text #00473c) */
#suggestions, .dark #suggestions {
    max-width: 800px !important;
    margin: 12px auto !important;
    display: flex !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
}

#chip, .dark #chip {
    background-color: #76f4e0 !important;
    background: #76f4e0 !important;
    border: none !important;
    color: #00473c !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 18px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
}

#chip:hover {
    background-color: #00bfa5 !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}

/* Floating Input Box Bar (bg white rounded-2xl shadow-xl) */
#input_row, .dark #input_row {
    max-width: 800px !important;
    margin: 0 auto !important;
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid rgba(108, 122, 118, 0.2) !important;
    border-radius: 20px !important;
    padding: 6px 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
    display: flex !important;
    align-items: center !important;
}

#msg_input, .dark #msg_input, .dark textarea {
    background-color: #ffffff !important;
    color: #161c27 !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 0.95rem !important;
}

#send_btn, .dark #send_btn {
    background-color: #00bfa5 !important;
    background: #00bfa5 !important;
    color: #00473c !important;
    border: none !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
}

#send_btn:hover {
    background-color: #00a896 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(0, 191, 165, 0.35) !important;
}

#upload_btn, .dark #upload_btn {
    background-color: transparent !important;
    border: none !important;
    font-size: 1.3rem !important;
    color: #3c4a46 !important;
}

/* Footer */
#footer, .dark #footer {
    max-width: 800px !important;
    margin: 16px auto 8px auto !important;
    color: #6c7a76 !important;
    font-size: 0.85rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    background-color: transparent !important;
}

/* Media Queries Responsive Desktop/Mobile */
@media (max-width: 1024px) {
    #sidebar {
        width: 100% !important;
        margin-bottom: 12px !important;
    }
}
"""

CUSTOM_CSS = MOCKUP_ZIP_CSS

def create_gradio_app():
    with gr.Blocks(title="DocBrain AI - Intelligent Document Analysis") as demo:
        # TopNavBar Header (Sticky)
        with gr.Row(elem_id="header"):
            with gr.Column(scale=2, min_width=200):
                gr.Markdown("## DocBrain AI", elem_id="header_logo")
            with gr.Column(scale=1, min_width=150):
                gr.Markdown(" **Guest User**")

        # Body Layout (Sidebar 280px + Main Chat Area)
        with gr.Row(equal_height=False):
            # SideNavBar Left
            with gr.Column(scale=1, min_width=260, elem_id="sidebar"):
                with gr.Column(elem_id="brand_card"):
                    gr.Markdown("### DocBrain AI", elem_id="brand_title")
                    gr.Markdown("<small>Intelligent Document Analysis</small>")

                new_session_btn = gr.Button("New Session", elem_id="new_session_btn", size="lg")

                gr.Markdown("---")
                initial_choices = [(sess["name"], sid) for sid, sess in reversed(list(sessions.items()))]
                session_radio = gr.Radio(
                    choices=initial_choices,
                    value=current_session_id,
                    label="Session List",
                    interactive=True
                )

                delete_session_btn = gr.Button("Delete Session", variant="stop", size="sm")

                gr.Markdown("---")
                top_k_slider = gr.Slider(
                    minimum=1, maximum=8, value=4, step=1,
                    label="Top-K Citations",
                    interactive=True
                )

                with gr.Row():
                    stats_btn = gr.Button("Stats", size="sm")
                    help_btn = gr.Button("Help", size="sm")

                stats_output = gr.Markdown(visible=False)

            # Main Chat Area
            with gr.Column(scale=3, min_width=500):
                init_sess = sessions.get(current_session_id, {})
                init_title = f"### {init_sess.get('name', 'Current Session')}"
                init_info = f" {init_sess.get('doc_count', 0)} documents  {init_sess.get('chunks', 0)} chunks"

                # Chat Header
                with gr.Row(elem_id="chat_header"):
                    session_title = gr.Markdown(init_title, elem_id="session_title")
                    doc_info = gr.Markdown(init_info, elem_id="doc_info_chip")

                # Messages Canvas
                chatbot = gr.Chatbot(value=init_sess.get("messages", []), label="", height=460, elem_id="chatbot")

                # Suggestion Chips
                with gr.Row(elem_id="suggestions"):
                    chip1 = gr.Button("Who is the Chairman?", size="sm", elem_id="chip")
                    chip2 = gr.Button("2025 Revenue?", size="sm", elem_id="chip")
                    chip3 = gr.Button("Current year GDP?", size="sm", elem_id="chip")

                # Chat Bar Floating Input Box
                with gr.Row(elem_id="input_row"):
                    upload_btn = gr.UploadButton("Upload", file_types=[".pdf", ".docx", ".txt"], scale=0, elem_id="upload_btn")
                    msg = gr.Textbox(
                        placeholder="Enter your question... (Press Enter to send)",
                        lines=1,
                        scale=5,
                        container=False,
                        elem_id="msg_input"
                    )
                    send_btn = gr.Button("Send", scale=0, elem_id="send_btn")

        # Footer
        with gr.Row(elem_id="footer"):
            gr.Markdown("v1.0.2  2024 DocBrain AI")
            gr.Markdown("GitHub &nbsp;&nbsp; Contact &nbsp;&nbsp; Privacy")

        # Logic handlers
        
        def new_session():
            global current_session_id
            new_num = len(sessions) + 1
            new_id = f"session_{new_num}_{int(time.time())}"
            sessions[new_id] = {
                "id": new_id,
                "name": f"Session {new_num}",
                "doc_count": 0,
                "chunks": 0,
                "documents": [],
                "messages": []
            }
            current_session_id = new_id
            save_sessions_to_disk(sessions)

            choices = [(sess["name"], sid) for sid, sess in reversed(list(sessions.items()))]
            return (
                gr.Radio(choices=choices, value=new_id),
                f"### {sessions[new_id]['name']}",
                " 0 documents  0 chunks",
                []
            )

        def switch_session(sid):
            global current_session_id
            if sid and sid in sessions:
                current_session_id = sid
                sess = sessions[sid]
                return (
                    f"### {sess['name']}",
                    f" {sess['doc_count']} documents  {sess['chunks']} chunks",
                    sess['messages']
                )
            return gr.update(), gr.update(), []

        def delete_session():
            global current_session_id
            if len(sessions) <= 1:
                return gr.update(), gr.update(), gr.update(), gr.update()

            pipeline = get_pipeline()
            pipeline.vector_store.delete_session_store(current_session_id)
            del sessions[current_session_id]

            first_id = list(sessions.keys())[0]
            current_session_id = first_id
            save_sessions_to_disk(sessions)

            choices = [(sess["name"], sid) for sid, sess in reversed(list(sessions.items()))]
            sess = sessions[first_id]
            return (
                gr.Radio(choices=choices, value=first_id),
                f"### {sess['name']}",
                f" {sess['doc_count']} documents  {sess['chunks']} chunks",
                sess['messages']
            )

        def upload_file_to_session(file_obj):
            global current_session_id
            if file_obj is None:
                return f" {sessions[current_session_id]['doc_count']} documents  {sessions[current_session_id]['chunks']} chunks", gr.update(), gr.update()

            pipeline = get_pipeline()
            filename = os.path.basename(file_obj.name)
            
            session_raw_dir = Path("data/sessions_raw") / current_session_id
            session_raw_dir.mkdir(parents=True, exist_ok=True)
            target_path = session_raw_dir / filename

            import shutil
            shutil.copy(file_obj.name, target_path)

            ingest_result = pipeline.process_and_ingest_file_for_session(current_session_id, str(target_path))

            if filename not in sessions[current_session_id]["documents"]:
                sessions[current_session_id]["documents"].append(filename)
                sessions[current_session_id]["doc_count"] = len(sessions[current_session_id]["documents"])

            sessions[current_session_id]["chunks"] = ingest_result.get("total_chunks", 0)

            if "Session" in sessions[current_session_id]["name"] or "Session" in sessions[current_session_id]["name"] or "Chưa có documents" in sessions[current_session_id]["name"]:
                sessions[current_session_id]["name"] = f"Session: {filename[:15]}"

            save_sessions_to_disk(sessions)

            choices = [(sess["name"], sid) for sid, sess in reversed(list(sessions.items()))]
            doc_info_text = f" {sessions[current_session_id]['doc_count']} documents  {sessions[current_session_id]['chunks']} chunks"
            title_text = f"### {sessions[current_session_id]['name']}"

            return doc_info_text, title_text, gr.Radio(choices=choices, value=current_session_id)

        def respond(message, history, top_k):
            global current_session_id
            if not message.strip():
                return history, ""

            pipeline = get_pipeline()
            session_doc_cnt = sessions[current_session_id]["doc_count"]

            result = pipeline.ask(
                query=message,
                session_id=current_session_id,
                session_doc_count=session_doc_cnt,
                top_k=top_k
            )

            answer = result["answer"]
            conf = int(result["confidence"] * 100)
            exec_time = result["execution_time_sec"]

            citations_str = ""
            if not result.get("is_conversational", False) and result["sources"]:
                citations_str += "\n\n**Source Citations:**\n"
                for src in result["sources"]:
                    snippet = src.get("text_snippet", "").replace("\n", " ").strip()
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    citations_str += f" **File**: `{src['source_file']}` (Page {src['page']})  Score: `{src.get('relevance_score', 0)}`\n"
                    citations_str += f"  > *\"{snippet}\"*\n"

            if result.get("is_conversational", False):
                full_bot_response = answer
            else:
                full_bot_response = f"{answer}{citations_str}"

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": full_bot_response})

            sessions[current_session_id]["messages"] = history

            save_sessions_to_disk(sessions)
            return history, ""

        def show_stats():
            active_cnt = len(sessions)
            total_chunks = sum(s.get("chunks", 0) for s in sessions.values())
            stats_md = f"### Vector Store Statistics:\n\n"
            stats_md += f"- **Total Sessions:** `{active_cnt}`\n"
            stats_md += f"- **Total Chunks:** `{total_chunks}`\n"
            return gr.update(visible=True, value=stats_md)

        # Event bindings
        new_session_btn.click(
            new_session,
            outputs=[session_radio, session_title, doc_info, chatbot]
        )

        session_radio.change(
            switch_session,
            inputs=[session_radio],
            outputs=[session_title, doc_info, chatbot]
        )

        delete_session_btn.click(
            delete_session,
            outputs=[session_radio, session_title, doc_info, chatbot]
        )

        send_btn.click(respond, [msg, chatbot, top_k_slider], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, top_k_slider], [chatbot, msg])

        upload_btn.upload(
            upload_file_to_session,
            inputs=[upload_btn],
            outputs=[doc_info, session_title, session_radio]
        )

        stats_btn.click(show_stats, outputs=[stats_output])
        help_btn.click(lambda: gr.update(visible=True, value="### Guide:\n1. Click New Session to create a new session.\n2. Click Upload để tải documents vào phiên.\n3. Đặt câu hỏi và bấm Send."), outputs=[stats_output])

        chip1.click(lambda: "Who is the Chairman?", outputs=[msg])
        chip2.click(lambda: "2025 Revenue?", outputs=[msg])
        chip3.click(lambda: "Current year GDP?", outputs=[msg])

    return demo
