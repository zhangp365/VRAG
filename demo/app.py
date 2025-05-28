import streamlit as st
from vrag_agent import VRAG
from PIL import Image
from time import sleep

# Set page configuration
st.set_page_config(
    page_title="VRAG: Discovering More in Depth",
    page_icon="🔍",  # Use a magnifying glass as the page icon
    layout="wide",   # Wide layout
    initial_sidebar_state="expanded"
)

def main():
    # Page title
    st.title("🔍 VRAG: Discovering More in Depth")
    st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 20px;
        }
        .header-text {
            font-size: 20px;
            color: #34495E;
            margin-bottom: 10px;
        }
        .sidebar-header {
            font-size: 18px;
            color: #1ABC9C;
            margin-bottom: 15px;
        }
        .submit-button {
            background-color: #2E86C1;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .divider {
            border-top: 2px solid #1ABC9C;
            margin-top: 5px;
            margin-bottom: 5px;
        }
        .success-message {
            background-color: #D5F5E3;
            color: #1E8449;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        /* Custom image style */
        .small-image img {
            max-width: auto; /* Maximum image width */
            max-height: 500px;     /* Maintain aspect ratio */
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize VRAG agent
    agent = VRAG()

    # Sidebar configuration
    with st.sidebar:
        st.markdown('<p class="sidebar-header">⚙️ Configuration Options</p>', unsafe_allow_html=True)
        MAX_ROUNDS = st.number_input('Number of Max Reasoning Iterations:', min_value=3, max_value=10, value=10, step=1)
        # n_max_doc = st.number_input('Maximum References to Use:', min_value=1, max_value=50, value=10, step=5)
        selected_example = st.selectbox(
            'Examples:', 
            ["助力证券公司上云的核心技术如何保证安全？", 
            "互联网IPV6活跃用户占比在2019年是多少，到2023年增加了多少？",
            # "高性能转发包含哪些技术？", 
            "What is the most commonly used travel app?", 
            # "Which country has the highest credit card penetration rate in Southeast Asia?",
            "What is the usage rate of travel agents for leisure in the country with the highest credit card penetration rate in Southeast Asia?"])
        st.markdown('<hr style="border:1px solid #1ABC9C">', unsafe_allow_html=True)  # Divider line

    # Question input box and button placed at the top of the page
    st.markdown('<p class="header-text">📝 Enter Your Question:</p>', unsafe_allow_html=True)

    agent.max_steps = MAX_ROUNDS
    question = st.text_input(
        "Question Input:",
        placeholder="Type your question here...",
        key="question_input",
        value=selected_example,  # Set the default question here
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1, 5])  # Adjust button layout
    with col1:
        submit_button = st.button("Submit", key="submit_button", help="Click to submit the question")
    with col2:
        if submit_button and question:
            st.markdown('<p class="success-message">Question submitted! Processing...</p>', unsafe_allow_html=True)

    # Create a scrollable container to display results
    result_container = st.container()
    with result_container:
        if submit_button and question:
            generator = agent.run(question)
            try:
                while True:
                    action, content, raw_content = next(generator)
                    if action == 'think':
                        st.markdown('<hr class="divider">', unsafe_allow_html=True)
                        st.info(f"💭 Thinking: {content}")
                    elif action == 'search':
                        st.info(f"🔍 Searching: {content}")
                    elif action == 'bbox':
                        st.info(f"📷 Region of Interest: {content}")
                    elif action == 'search_image':
                        st.markdown('<div class="small-image">', unsafe_allow_html=True)  # Apply custom styles
                        st.image(content, caption='Retrieved Image', width=300)  # Set image width to 300 pixels
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif action == 'crop_image':
                        # Use two-column layout to display original and cropped images
                        col1, col3, col2 = st.columns(3)
                        with col1:
                            st.markdown('<div class="small-image">', unsafe_allow_html=True)
                            st.image(raw_content, caption='Original Image', width=300)  # Set image width to 300 pixels
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="small-image">', unsafe_allow_html=True)
                            st.image(Image.open('./assets/crop.png'), caption='Cropped Image', width=500) 
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="small-image">', unsafe_allow_html=True)
                            st.image(content, caption='Cropped Image', width=300)  # Set image width to 300 pixels
                            st.markdown('</div>', unsafe_allow_html=True)
            except StopIteration as e:
                action, content, raw_response = e.value
                if action == 'answer':
                    st.success(f"✅ Answer: {content}")

if __name__ == "__main__":
    main()
