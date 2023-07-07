import streamlit as st
import pandas as pd
from numpy import random

import torch
from torch import cuda

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from get_description import get_description


@st.cache_resource
def load_model(device):
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    pipeline = pipeline.to(device)
    pipeline.enable_xformers_memory_efficient_attention()

    return pipeline


@st.cache_resource
def noise(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def main():
    st.title("앨범 만들기")

    # 1. Input album information
    st.header("앨범 정보 입력하기")

    song_names = st.text_input("노래 제목을 입력해주세요.", "퀸카 (Queencard)")
    artist_name = st.text_input("가수 이름을 입력해주세요.", "(여자)아이들")
    genre_kr = st.selectbox(
        "장르를 선택해주세요.",
        ["발라드", "댄스", "랩/힙합", "R&B/Soul", "인디음악", "록/메탈", "트로트", "포크/블루스"],
    )
    album_name = st.text_input("앨범 이름을 입력해주세요.", "I feel")
    release = st.date_input("날짜를 선태해주세요.")
    lyric = st.text_area("가사를 입력해주세요.")

    year = release.year
    month = release.month
    day = release.day
    # version - 1
    season = f"{year}-{month}-{day}"
    # version - 2
    if month > 11 or month < 3:
        season = "winter"
    elif 3 <= month < 6:
        season = "spring"
    elif 6 <= month < 9:
        season = "summer"
    elif 9 <= month < 12:
        season = "fall"

    info = {
        "노래 제목": song_names,
        "가수 이름": artist_name,
        "노래 장르": genre_kr,
        "앨범 이름": album_name,
        "발매일": f"{year}년 {month}월 {day}일({season})",
        "가사": lyric,
    }

    info_df = pd.DataFrame(list(info.values()), index=list(info.keys()), columns=["입력"])

    info_table = st.dataframe(info_df, use_container_width=True)

    # Translation
    genre_t = {
        "발라드": "ballad",
        "댄스": "dance",
        "랩/힙합": "hip-hop",
        "R&B/Soul": "R&B",
        "인디음악": "Indie",
        "록/메탈": "rock",
        "트로트": "trot",
        "포크/블루스": "fork",
    }
    genre = genre_t[genre_kr]

    # 2. Make prompt
    st.session_state["prompt"] = ""

    # 3. Inference
    st.header("이미지 생성 하기")
    if st.button("이미지 생성 하기", use_container_width=True):
        description = get_description(
            lyric, artist_name, album_name, season, song_names
        )
        st.session_state[
            "prompt"
        ] = f"A photo of a {genre} album cover with a {description} atmosphere visualized."

        with st.spinner("Wait for it..."):
            device = "cuda" if cuda.is_available() else "cpu"
            pipeline = load_model(device)

            images = []
            captions = []
            for seed in random.choice(range(0, 100), size=4, replace=False):
                generator = noise(device, seed)
                with torch.autocast(device):
                    image = pipeline(
                        prompt=st.session_state["prompt"],
                        num_inference_steps=20,
                        generator=generator,
                    ).images[
                        0
                    ]  # Default: 50
                images.append(image)
                captions.append(f"{st.session_state['prompt']} - {int(seed)}")

            st.image(images, caption=captions)


if __name__ == "__main__":
    main()
