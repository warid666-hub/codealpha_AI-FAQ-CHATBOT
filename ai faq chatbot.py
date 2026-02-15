import tkinter as tk
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer, util
import torch



faq_data = [
    {"question": "What is your refund policy?",
     "answer": "You can request a refund within 30 days."},

    {"question": "How do I reset my password?",
     "answer": "Click on 'Forgot Password' on the login page."},

    {"question": "Do you provide international shipping?",
     "answer": "Yes, we ship worldwide."},

    {"question": "Can you deliver to my location?",
     "answer": "Yes, we deliver worldwide."},

    {"question": "What payment methods are accepted?",
     "answer": "We accept Visa, Mastercard, and PayPal."},

    {"question": "Can I change my order after placing it?",
     "answer": "Yes, you can modify your order within 2 hours of placement."}
]



print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

faq_questions = [item["question"] for item in faq_data]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

print("Model loaded successfully!")

def get_answer(user_question):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, faq_embeddings)
    best_match_idx = torch.argmax(cosine_scores).item()
    return faq_data[best_match_idx]["answer"]


root = tk.Tk()
root.title("FUTURISTIC FAQ CHATBOT")
root.geometry("900x650")
root.configure(bg="#1a102b")
root.resizable(True, True)

BG_MAIN = "#1a102b"
GLASS_BG = "#231942"
USER_COLOR = "#9d4edd"
BOT_COLOR = "#3c096c"
ACCENT = "#c77dff"
TEXT_COLOR = "#f8f9fa"

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)


title = tk.Label(
    root,
    text="FURIISTIC FAQ CHATBOT",
    font=("Segoe UI", 24, "bold"),
    bg=BG_MAIN,
    fg=ACCENT
)
title.grid(row=0, column=0, pady=25)


chat_frame = tk.Frame(root, bg=GLASS_BG, bd=0)
chat_frame.grid(row=1, column=0, padx=40, pady=10, sticky="nsew")

chat_frame.grid_rowconfigure(0, weight=1)
chat_frame.grid_columnconfigure(0, weight=1)

chat_log = scrolledtext.ScrolledText(
    chat_frame,
    wrap=tk.WORD,
    font=("Segoe UI", 12),
    bg=GLASS_BG,
    fg=TEXT_COLOR,
    insertbackground="white",
    bd=0,
    padx=15,
    pady=15
)
chat_log.grid(row=0, column=0, sticky="nsew")
chat_log.config(state="disabled")


chat_log.tag_config("user_label", foreground="#e0aaff", font=("Segoe UI", 10, "bold"))
chat_log.tag_config("bot_label", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
chat_log.tag_config("user_msg", background=USER_COLOR, foreground="white",
                    lmargin1=80, lmargin2=80, spacing3=8)
chat_log.tag_config("bot_msg", background=BOT_COLOR, foreground="white",
                    lmargin1=20, lmargin2=20, spacing3=8)

def add_message(sender, message):
    chat_log.config(state="normal")

    if sender == "You":
        chat_log.insert(tk.END, "\nYou\n", ("user_label",))
        chat_log.insert(tk.END, message + "\n", ("user_msg",))
    else:
        chat_log.insert(tk.END, "\nAI\n", ("bot_label",))
        chat_log.insert(tk.END, message + "\n", ("bot_msg",))

    chat_log.config(state="disabled")
    chat_log.see(tk.END)

input_frame = tk.Frame(root, bg=BG_MAIN)
input_frame.grid(row=2, column=0, padx=40, pady=20, sticky="ew")

input_frame.grid_columnconfigure(0, weight=1)

user_entry = tk.Entry(
    input_frame,
    font=("Segoe UI", 14),
    bg="#2e1a47",
    fg="white",
    insertbackground="white",
    bd=0
)
user_entry.grid(row=0, column=0, sticky="ew", ipady=10, padx=(0, 15))

def send_message():
    question = user_entry.get().strip()
    if not question:
        return

    add_message("You", question)
    answer = get_answer(question)
    add_message("AI", answer)
    user_entry.delete(0, tk.END)

send_btn = tk.Button(
    input_frame,
    text="SEND",
    font=("Segoe UI", 12, "bold"),
    bg=ACCENT,
    fg="black",
    bd=0,
    padx=20,
    pady=8,
    command=send_message
)
send_btn.grid(row=0, column=1)


def on_enter(e):
    send_btn.config(bg="#e0aaff")

def on_leave(e):
    send_btn.config(bg=ACCENT)

send_btn.bind("<Enter>", on_enter)
send_btn.bind("<Leave>", on_leave)

root.bind("<Return>", lambda event: send_message())

root.mainloop()
