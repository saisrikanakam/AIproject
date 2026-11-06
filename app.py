
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import joblib
import streamlit as st


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "edubot_progress.db"

st.set_page_config(
    page_title="EduBot | AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. DARK THEME
# ==================================================

st.markdown("""
<style>
.stApp {
    background: #0D111B;
    color: #E7EAF4;
}
[data-testid="stAppViewContainer"] {
    background: #0D111B;
}
.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #E5E1FF !important;
}
p, li, label {
    color: #DCE1EF;
}
[data-testid="stSidebar"] {
    background: #151A29;
    border-right: 1px solid #292F43;
}
[data-testid="stChatMessage"],
[data-testid="stMetric"],
[data-testid="stForm"] {
    background: #181E2C;
    border: 1px solid #293147;
    border-radius: 14px;
    padding: 16px;
}
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #745CF5, #477FF0);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 650;
}
[data-baseweb="select"] > div,
.stTextInput input,
.stTextArea textarea {
    background: #181E2C !important;
    color: #F5F5FC !important;
    border-color: #343D56 !important;
}
hr {
    border-color: #293147 !important;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. DATABASE
# ==================================================

def init_progress_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id TEXT NOT NULL,
                topic_title TEXT NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                percentage REAL NOT NULL,
                attempted_at TEXT NOT NULL
            )
        """)


def save_quiz_result(topic, score, total):
    percentage = score / total * 100 if total else 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO quiz_history
            (topic_id, topic_title, score, total,
             percentage, attempted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(topic["id"]),
            topic["title"],
            score,
            total,
            percentage,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))


def load_quiz_history():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT topic_id, topic_title, score, total,
                   percentage, attempted_at
            FROM quiz_history
            ORDER BY id DESC
        """).fetchall()


init_progress_db()


# ==================================================
# 4. LOAD KNOWLEDGE BASE AND MODEL
# ==================================================

@st.cache_data
def load_knowledge():
    path = BASE_DIR / "knowledge_base.json"

    with open(path, encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_model():
    path = BASE_DIR / "intent_model.joblib"
    return joblib.load(path)


try:
    knowledge = load_knowledge()
    topics = knowledge.get("topics", [])
    model = load_model()
except Exception as error:
    st.error(
        "EduBot could not load its knowledge base or intent model. "
        "Check that knowledge_base.json and intent_model.joblib "
        f"exist beside app.py. Details: {error}"
    )
    st.stop()


if not topics:
    st.error("No topics were found in knowledge_base.json.")
    st.stop()


# Validate required topic fields.
valid_topics = [
    topic for topic in topics
    if all(key in topic for key in ("id", "title"))
]

if not valid_topics:
    st.error("Your knowledge base needs topics with both id and title.")
    st.stop()

topics = valid_topics
topic_titles = [topic["title"] for topic in topics]


# ==================================================
# 5. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Welcome to EduBot! 👋\n\n"
            "Ask a question, request revision notes, "
            "or start a quiz.\n\n"
            "Try: **What is Photosynthesis?**"
        ),
    }]

# The quiz selector stores a topic TITLE.
if "quiz_topic_selector" not in st.session_state:
    st.session_state["quiz_topic_selector"] = topic_titles[0]

if st.session_state["quiz_topic_selector"] not in topic_titles:
    st.session_state["quiz_topic_selector"] = topic_titles[0]


# ==================================================
# 6. FIND TOPIC
# ==================================================

def find_topic(message):
    query = message.lower().strip()
    matches = []

    for topic in topics:
        terms = topic.get("keywords", []) + [topic["title"]]

        for term in terms:
            term = str(term).lower().strip()

            if term and term in query:
                matches.append((len(term), topic))

    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    return None


# ==================================================
# 7. INTENT DETECTION
# ==================================================

def predict_intent(message):
    try:
        probabilities = model.predict_proba([message])[0]
        best_index = probabilities.argmax()

        if probabilities[best_index] < 0.38:
            return "unknown"

        return model.classes_[best_index]

    except Exception:
        return "unknown"


# ==================================================
# 8. GENERATE RESPONSES
# ==================================================

def get_response(message):
    message_lower = message.lower().strip()
    intent = predict_intent(message)
    topic = find_topic(message)

    # Help
    if message_lower in ["help", "hi", "hello", "hey"]:
        return (
            "## 👋 Welcome to EduBot!\n\n"
            "Try these commands:\n\n"
            "- 📚 Give me detailed notes on Photosynthesis\n"
            "- 🔬 Explain Gravity in detail\n"
            "- 📝 Start a quiz on Electricity\n"
            "- 💡 Explain Quadratic Equations\n\n"
            "Type a topic or ask me a question!"
        )

    if intent == "greeting":
        return (
            "Hello! 👋 Welcome to EduBot!\n\n"
            "Ask me a question, request revision notes, "
            "or start a quiz."
        )

    if intent == "help":
        available = ", ".join(topic_titles)
        return (
            "📚 I can answer supported questions, provide notes, "
            "and conduct quizzes.\n\n"
            f"**Available topics:** {available}"
        )

    # IMPORTANT: Handle quiz commands BEFORE notes.
    quiz_requested = (
        "quiz" in message_lower
        and any(
            word in message_lower
            for word in ["start", "begin", "take", "give", "create"]
        )
    ) or intent == "start_quiz"

    if quiz_requested:
        if topic:
            # This is the same key used by the quiz selectbox.
            st.session_state["quiz_topic_selector"] = topic["title"]

            return (
                f"📝 Your **{topic['title']}** quiz is ready!\n\n"
                "I've selected that topic in Quiz & Assessment below. "
                "Scroll down to answer the questions. 🎯"
            )

        return (
            "Which topic would you like to take a quiz on? 😊\n\n"
            f"**Available topics:** {', '.join(topic_titles)}\n\n"
            "Example: Start a quiz on Electricity"
        )

    # Detailed notes
    notes_keywords = [
        "notes",
        "revision",
        "explain",
        "in detail",
        "detailed",
        "study material",
        "teach me",
        "full explanation",
    ]

    wants_notes = any(
        keyword in message_lower for keyword in notes_keywords
    )

    if wants_notes:
        if topic:
            return (
                f"## 📚 {topic['title']} — Detailed Revision Notes\n\n"
                f"{topic.get('notes', 'Notes are not available yet.')}"
            )

        return (
            "Which topic would you like notes on? 😊\n\n"
            f"**Available topics:** {', '.join(topic_titles)}"
        )

    # Recognized topic question
    if topic:
        answer = topic.get("answer", topic.get("notes", ""))
        return f"**{topic['title']}**\n\n{answer}"

    if intent == "ask_question":
        return (
            "I don't have that topic in my knowledge base yet. 😊\n\n"
            f"Try one of these topics: {', '.join(topic_titles)}"
        )

    return (
        "I'm not sure I understood that. 🤔\n\n"
        "Try asking a supported question, requesting notes, "
        "starting a quiz, or typing 'help'."
    )


# ==================================================
# 9. APP HEADER
# ==================================================

st.markdown("""
<div style="
    background: linear-gradient(135deg, #191E35, #252044);
    border: 1px solid #37345D;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;">
    <div style="color:#A99BFF;font-weight:700;letter-spacing:2px;">
        YOUR AI LEARNING COMPANION
    </div>
    <h1 style="margin:10px 0;color:white !important;">
        🎓 EduBot
    </h1>
    <p style="color:#C4C8DF;margin-bottom:0;">
        Learn smarter. Ask questions. Test your knowledge.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 10. STUDENT DASHBOARD
# ==================================================

st.markdown("## 📊 Student Dashboard")
st.caption("Track quiz history and find topics to revise.")

history = load_quiz_history()

if history:
    attempts = len(history)
    avg_score = sum(row[4] for row in history) / attempts
    best_score = max(row[4] for row in history)
    unique_topics = len(set(row[0] for row in history))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quizzes Completed", attempts)
    c2.metric("Average Score", f"{avg_score:.1f}%")
    c3.metric("Best Score", f"{best_score:.1f}%")
    c4.metric("Topics Practised", unique_topics)

    st.progress(min(avg_score / 100, 1.0))

    st.markdown("### 📈 Quiz History")

    st.dataframe(
        [
            {
                "Topic": row[1],
                "Score": f"{row[2]}/{row[3]}",
                "Percentage": f"{row[4]:.1f}%",
                "Attempted At": row[5],
            }
            for row in history
        ],
        use_container_width=True,
        hide_index=True,
    )

    topic_scores = {}

    for row in history:
        topic_id, title, score, total, percentage, attempted_at = row

        if topic_id not in topic_scores:
            topic_scores[topic_id] = {
                "title": title,
                "scores": [],
            }

        topic_scores[topic_id]["scores"].append(percentage)

    weak_topics = []

    for item in topic_scores.values():
        average = sum(item["scores"]) / len(item["scores"])
        if average < 60:
            weak_topics.append({
                "title": item["title"],
                "average": average,
            })

    weak_topics.sort(key=lambda item: item["average"])

    st.markdown("### 🎯 Topics to Improve")

    if weak_topics:
        for item in weak_topics:
            st.warning(
                f"**{item['title']}** — "
                f"Average score: {item['average']:.1f}%"
            )

        st.markdown("### 💡 Recommended Revision")

        for item in weak_topics:
            revision_topic = next(
                (
                    t for t in topics
                    if t["title"] == item["title"]
                ),
                None,
            )

            if revision_topic:
                with st.expander(
                    f"Revise {item['title']} "
                    f"({item['average']:.1f}%)"
                ):
                    st.markdown(
                        revision_topic.get("notes", "Notes unavailable.")
                    )

                    for resource in revision_topic.get("resources", []):
                        st.markdown(f"- {resource}")
    else:
        st.success("No weak topics identified. Keep practising! 🎉")

else:
    st.info(
        "👋 Complete your first quiz to unlock your learning "
        "progress and personalised revision suggestions."
    )

st.divider()


# ==================================================
# 11. SIDEBAR LEARNING RESOURCES
# ==================================================

with st.sidebar:
    st.markdown("## 📚 Learning Resources")
    st.caption("Explore your study materials.")

    subjects = sorted(set(
        topic.get("subject", "General") for topic in topics
    ))

    selected_subject = st.selectbox(
        "Choose a subject",
        subjects,
        key="sidebar_subject",
    )

    subject_topics = [
        topic for topic in topics
        if topic.get("subject", "General") == selected_subject
    ]

    if subject_topics:
        selected_title = st.selectbox(
            "Select topic",
            [topic["title"] for topic in subject_topics],
            key="sidebar_topic",
        )

        selected_topic = next(
            topic for topic in subject_topics
            if topic["title"] == selected_title
        )

        st.divider()
        st.markdown("### 📝 Revision Notes")
        st.markdown(
            selected_topic.get("notes", "Notes unavailable.")
        )

        st.divider()
        st.markdown("### 💡 Suggested Activities")

        for resource in selected_topic.get("resources", []):
            st.markdown(f"- {resource}")

    st.divider()
    st.caption("Keep learning. One topic at a time. 💜")


# ==================================================
# 12. CHAT INTERFACE
# ==================================================

st.markdown("## 💬 Ask EduBot")
st.caption("Ask questions, request notes, or start a topic quiz.")

for chat_message in st.session_state.messages:
    with st.chat_message(chat_message["role"]):
        st.markdown(chat_message["content"])

with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input(
        "Your question",
        placeholder="e.g. Start a quiz on Electricity",
        label_visibility="collapsed",
    )

    chat_submitted = st.form_submit_button(
        "Send message ➤",
        use_container_width=True,
    )

if chat_submitted and prompt.strip():
    user_message = prompt.strip()

    st.session_state.messages.append({
        "role": "user",
        "content": user_message,
    })

    response = get_response(user_message)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })

    # Rerun so the quiz selector reflects the requested topic.
    st.rerun()


# ==================================================
# 13. QUIZ AND ASSESSMENT
# ==================================================

st.divider()
st.markdown("## 📝 Quiz & Assessment")
st.caption("Test your understanding and review your answers.")

if topics:
    # IMPORTANT:
    # The selector uses topic titles as options.
    # The chat command updates this same session-state key.
    chosen_title = st.selectbox(
        "Choose your quiz topic",
        topic_titles,
        key="quiz_topic_selector",
    )

    chosen_topic = next(
        topic for topic in topics
        if topic["title"] == chosen_title
    )

    quiz_questions = chosen_topic.get("quiz", [])

    if quiz_questions:
        st.markdown(f"### 🎯 Topic: {chosen_topic['title']}")
        st.caption(
            f"{len(quiz_questions)} questions · "
            "Choose the best answer for each question."
        )

        with st.form("quiz_form"):
            answers = {}

            for i, question in enumerate(quiz_questions):
                st.markdown(
                    f"**Q{i + 1}. {question['question']}**"
                )

                options = question["options"]

                answers[i] = st.radio(
                    "Choose your answer",
                    range(len(options)),
                    format_func=lambda n, opts=options: opts[n],
                    key=f"{chosen_topic['id']}_q{i}",
                )

                if i < len(quiz_questions) - 1:
                    st.markdown("---")

            quiz_submitted = st.form_submit_button(
                "✅ Submit Quiz",
                use_container_width=True,
            )

        if quiz_submitted:
            score = sum(
                answers[i] == question["answer"]
                for i, question in enumerate(quiz_questions)
            )

            total = len(quiz_questions)
            percentage = score / total * 100 if total else 0

            save_quiz_result(chosen_topic, score, total)

            st.divider()
            st.markdown("### 🏆 Your Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{score}/{total}")
            c2.metric("Percentage", f"{percentage:.0f}%")
            c3.metric("Correct Answers", f"{score} of {total}")

            st.progress(int(percentage))

            if percentage == 100:
                st.success("Excellent! 🎉 Every answer is correct!")
            elif percentage >= 50:
                st.info("Good work! Review the explanations to improve.")
            else:
                st.warning("Keep practising! Review the notes and try again.")

            st.markdown("### 📖 Answer Review")

            for i, question in enumerate(quiz_questions):
                correct = question["answer"]

                if answers[i] == correct:
                    st.success(f"Question {i + 1}: Correct!")
                else:
                    st.error(
                        f"Question {i + 1}: Correct answer — "
                        f"{question['options'][correct]}"
                    )

                st.caption(
                    question.get("explanation", "No explanation available.")
                )

            if percentage < 100:
                st.markdown("### 💡 Recommended Revision")

                for resource in chosen_topic.get("resources", []):
                    st.markdown(f"- {resource}")

    else:
        st.info(
            f"No quiz questions are available for {chosen_topic['title']} yet."
        )


# ==================================================
# 14. FOOTER
# ==================================================

st.divider()
st.caption("🎓 EduBot | AI-powered educational learning prototype")
