import streamlit as st

def show_top_nav(pages):
    """Professional, emoji-free header navigation with perfectly aligned and sized buttons."""

    # Ensure the current page is set in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = pages[0]
    current_index = pages.index(st.session_state.current_page)

    # CSS: Tight and aligned
    st.markdown("""
    <style>
    .stButton > button {
        width: 100% !important;
        min-width: 90px;
        max-width: 180px;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        border: none;
        margin: 0 0 0 0;
        box-shadow: none !important;
    }
    .stButton > button:disabled {
        opacity: 0.54 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #eee !important;
    }
    /* Selectbox same height, border */
    section[data-testid="stSelectbox"] > div {
        min-height: 40px !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # No emojis in nav
    clean_pages = [p.encode('ascii', 'ignore').decode().strip() for p in pages]

    # Lay out: wide center, tight buttons
    col1, col2, col3 = st.columns([1, 7, 1], gap="small")

    with col1:
        prev_clicked = st.button("Previous", disabled=(current_index == 0), use_container_width=True, key="prev_btn")
    with col2:
        selected = st.selectbox(
            "Go to Page",
            clean_pages,
            index=current_index,
            key="main_nav_select",
            label_visibility="collapsed"
        )
    with col3:
        next_clicked = st.button("Next", disabled=(current_index == len(clean_pages)-1), use_container_width=True, key="next_btn")

    # Logic (handle first/last page, no double triggers)
    if prev_clicked and current_index > 0:
        st.session_state.current_page = pages[current_index - 1]
        st.rerun()
    if next_clicked and current_index < len(clean_pages) - 1:
        st.session_state.current_page = pages[current_index + 1]
        st.rerun()
    if selected != clean_pages[current_index]:
        st.session_state.current_page = pages[clean_pages.index(selected)]
        st.rerun()

    st.write("---")  # Divider
    return st.session_state.current_page
