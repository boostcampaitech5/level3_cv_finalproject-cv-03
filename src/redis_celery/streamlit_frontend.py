# Python built-in modules
import base64
import io
from io import BytesIO

# Frontend
import streamlit as st

# Other modules
import pandas as pd
import requests
from PIL import Image

# Built-in modules
try:
    from src.redis_celery.utils import load_yaml
except:
    from utils import load_yaml


def main():
    # Load config
    request_config = load_yaml(
        os.path.join("src/redis_celery/config", "private.yaml"), "request"
    )
    public_config = load_yaml(os.path.join("src/redis_celery/config", "public.yaml"))
    language = public_config["language"]
    translation_config = load_yaml(
        os.path.join("src/redis_celery/config", "translation.yaml"), language
    )

    # Frontend
    st.title(translation_config["title"])

    # 1. Input album information
    st.header(translation_config["album_info"])

    song_names = st.text_input(
        translation_config["song_names"]["text"],
        placeholder=translation_config["song_names"]["placeholder"],
    )
    artist_name = st.text_input(
        translation_config["artist_name"]["text"],
        placeholder=translation_config["artist_name"]["placeholder"],
    )
    genre = st.selectbox(
        translation_config["genre"]["text"],
        translation_config["genre"]["list"],
    )
    album_name = st.text_input(
        translation_config["album_name"]["text"],
        placeholder=translation_config["album_name"]["placeholder"],
    )
    lyric = st.text_area(
        translation_config["lyric"]["text"],
        placeholder=translation_config["lyric"]["placeholder"],
    )

    info = {
        translation_config["info"][0]: song_names,
        translation_config["info"][1]: artist_name,
        translation_config["info"][2]: genre,
        translation_config["info"][3]: album_name,
        translation_config["info"][4]: lyric,
    }

    # 2. Show info dataframe
    info_df = pd.DataFrame(
        list(info.values()),
        index=list(info.keys()),
        columns=[translation_config["dataframe"]["col"]],
    )
    info_table = st.dataframe(info_df, use_container_width=True)

    # 3. Inference
    request_info = {
        request_config["info"][0]: song_names,
        request_config["info"][1]: artist_name,
        request_config["info"][2]: genre,
        request_config["info"][3]: album_name,
        request_config["info"][4]: lyric,
    }

    st.header(translation_config["inference"]["header"])
    gen_button = st.button(
        translation_config["inference"]["button_message"], use_container_width=True
    )

    # Set 'img' variable to session state
    if "img" not in st.session_state:
        st.session_state.img = ["", "", "", ""]

    # Create 2x2 grid
    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2]

    if gen_button:
        with st.spinner(translation_config["inference"]["wait_message"]):
            # Call the FastAPI server to generate the album cover
            response = requests.post(request_config["gen_address"], json=request_info)
            if response.status_code == 200:
                image_urls = response.json()["images"]

                # Assign images to the cells in the grid
                for i, url in enumerate(image_urls):
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content))
                    st.session_state.img[i] = img
                    cols[i].image(img, width=300)
            else:
                st.error(translation_config["inference"]["fail_meesage"]["generate"])
    else:
        if st.session_state.img == ["", "", "", ""]:
            for i in range(len(cols)):
                cols[i].empty()
        else:
            for i in range(len(cols)):
                cols[i].image(st.session_state.img[i], width=300)

    # 4. Review
    with st.expander(translation_config["review"]["expander"]):
        review_rating = st.select_slider(
            translation_config["review"]["rating"], options=range(1, 6), value=5
        )
        st.markdown(
            "<h5 style='text-align: center;'>"
            + "❤️" * review_rating
            + "🖤" * (5 - review_rating)
            + "</h5>",
            unsafe_allow_html=True,
        )
        review_text = st.text_input(
            translation_config["review"]["review_text"]["text"],
            placeholder=translation_config["review"]["review_text"]["placeholder"],
            label_visibility="hidden",
        )
        review_btn = st.button(translation_config["review"]["button"])
        if review_btn:
            review = {"rating": review_rating, "comment": review_text}
            response = requests.post(request_config["review_address"], json=review)
            if response.status_code == 200:
                st.write(translation_config["review"]["message"]["success"])
            else:
                st.error(translation_config["review"]["message"]["fail"])


if __name__ == "__main__":
    main()
