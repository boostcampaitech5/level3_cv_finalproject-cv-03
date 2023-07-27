// 페이지 로드 시 로그인 상태 확인
window.onload = function () {
    updateLoginState();
};

let user_nickname;
let user_email;
//카카오 로그인
function kakaoLogin() {
    Kakao.Auth.login({
        success: function (response) {
            Kakao.API.request({
                url: '/v2/user/me',
                success: function (response) {
                    // alert('사용자 닉네임: ' + response.kakao_account.profile.nickname + '\n'
                    //     + '사용자 성별: ' + response.kakao_account.gender + '\n'
                    //     + '사용자 연령대: ' + response.kakao_account.age_range + '\n'
                    //     + '사용자 이메일: ' + response.kakao_account.email);
                    user_nickname = response.kakao_account.profile.nickname
                    user_email = response.kakao_account.email
                    sessionStorage.setItem('isLoggedIn', 'true');
                    sessionStorage.setItem('user_nickname', user_nickname);
                    sessionStorage.setItem('user_email', user_email);   // 테이블에서 사용자 식별할때는 이메일로 해야할 듯
                    updateLoginState();
                },
                fail: function (error) {
                    alert(
                        'login success, but failed to request user information: ' +
                        JSON.stringify(error)
                    )
                },
            })
        },
        fail: function (error) {
            console.log(error)
        },
    })
}

//카카오 로그아웃
function kakaoLogout() {
    if (Kakao.Auth.getAccessToken()) {
        Kakao.API.request({
            url: '/v1/user/unlink',
            success: function (response) {
                alert('로그아웃되었습니다.')
                sessionStorage.removeItem('isLoggedIn');
                sessionStorage.removeItem('user_nickname');
                sessionStorage.removeItem('user_email');
                updateLoginState();
            },
            fail: function (error) {
                alert('fail: ' + JSON.stringify(error))
            },
        })
        Kakao.Auth.setAccessToken(undefined)
    }
    else {
        alert('로그인 상태가 아닙니다.')
    }
}
// 로그인 상태에 따라 화면 갱신
function updateLoginState() {
    const isLoggedIn = sessionStorage.getItem('isLoggedIn');
    // alert(isLoggedIn)
    if (isLoggedIn === 'true') {
        // 로그인 상태
        document.getElementById('kakao_login').style.display = 'none';
        document.getElementById('mypage_login').style.display = 'block';
        document.getElementById('kakao_logout').style.display = 'block';
        document.getElementById('text_login').innerText = sessionStorage.getItem('user_nickname') + '님'
    } else {
        // 로그아웃 상태
        document.getElementById('kakao_login').style.display = 'block';
        document.getElementById('mypage_login').style.display = 'none';
        document.getElementById('kakao_logout').style.display = 'none';
        document.getElementById('text_login').innerText = '';
    }
}

function imageDownload(num) {
    new_a = document.createElement("a");
    new_a.href = document.getElementById("image" + num).src;
    new_a.download = "album_image_" + num + ".jpeg";
    document.body.appendChild(new_a);
    new_a.click();
    document.body.removeChild(new_a);
}

function selectGenre(genre) {
    genreItems.forEach(item => {
        item.classList.remove('selected');
    });
    genre.classList.add('selected');
}

document.addEventListener("DOMContentLoaded", () => {
    // loginUser()
    select_model = document.querySelectorAll('.select_model')
    model2_contents = document.querySelectorAll('.model_content')
    select_model.forEach((radio) => {
        radio.addEventListener('change', () => {
            if (radio.checked) {
                if (radio.id == "listGroupRadioGrid2") {
                    document.getElementById('model2_content').style.display = "block"
                }
                else {
                    document.getElementById('model2_content').style.display = "none"
                }
            }
        })
    })

    document.querySelector('#kakao_login').addEventListener('click', () => {
        kakaoLogin();
    })
    document.querySelector('#imageUpload').addEventListener('change', function () {
        const selectedFiles = this.files;
        const previewContainer = document.querySelector('#imagePreview');

        // 이미지 갯수가 최소 4장, 최대 10장인지 확인
        if (selectedFiles.length < 4 || selectedFiles.length > 10) {
            alert('최소 4장에서 최대 10장의 이미지를 업로드해주세요.');
            return;
        }

        // 기존의 미리보기 이미지를 모두 삭제
        while (previewContainer.firstChild) {
            previewContainer.removeChild(previewContainer.firstChild);
        }

        // 선택된 파일들의 미리보기 이미지를 생성하여 추가
        for (let i = 0; i < selectedFiles.length; i++) {
            const file = selectedFiles[i];
            const reader = new FileReader();

            reader.onload = function (event) {
                const image = document.createElement('img');
                image.setAttribute('src', event.target.result);
                image.setAttribute('class', 'preview-image');
                previewContainer.appendChild(image);
            };

            reader.readAsDataURL(file);
        }
    });


    // 모달을 열 때 클릭된 버튼 숫자정보(1~4) 가져오기
    $('#download_modal').on('show.bs.modal', function (event) {
        button = $(event.relatedTarget);
        buttonType = button.data('button-type');
    });

    document.querySelector("#review_send_btn").addEventListener("click", async (e) => {
        // e.preventDefault()
        user_starpoint = 0
        user_review = document.getElementById("review_comment").value
        for (let i = 1; i <= 10; i++) {
            cur_review = "starpoint_" + i
            if (document.getElementById(cur_review).checked == true) {
                user_starpoint = i / 2;
                break;
            }
        }
        if (user_starpoint == 0 || user_review == "") {
            alert('별점과 리뷰를 모두 작성해주세요.');
            document.getElementById("review_comment").focus();
        }
        else {
            const reviewData = {
                rating: user_starpoint,
                comment: user_review
            };

            try {
                const response = await fetch('http://localhost:8000/review', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(reviewData),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                console.log(data);  // Log the response data to check if it's correct

                $('#download_modal').modal('hide');
                // TODO : 리뷰 저장하는 코드 추가
                alert("소중한 리뷰 감사드립니다!" + "\n" + "별점 : " + user_starpoint + "점" + "\n" + "한줄평 : " + user_review);
                // 리뷰 초기화
                document.getElementById("review_comment").value = ""
                document.getElementById(cur_review).checked = false

                // 리뷰작성시 이미지 다운로드
                imageDownload(buttonType);
            } catch (error) {
                console.error('Error:', error);
            }
        }
    })

    const badge_checked = "badge bg-primary-subtle border border-primary-subtle text-primary-emphasis rounded-pill mb-1 genre"
    const badge_not_checked = "badge bg-secondary-subtle border border-secondary-subtle text-secondary-emphasis rounded-pill mb-1 genre"
    document.querySelector("#img_create_btn").addEventListener("click", async (e) => {
        e.preventDefault()
        // 버튼 동작 체크
        console.log("Button Clicked!");

        const required_ids = ["song_name", "artist_name", "album_name"];
        for (let i of required_ids) {
            if (document.getElementById(i).value == "") {
                alert("모든 필수 입력란을 입력해주세요.");
                document.getElementById(i).focus();
                return;
            }
        }
        select_song = document.getElementById("song_name").value
        select_artist = document.getElementById("artist_name").value
        select_album = document.getElementById("album_name").value
        select_lyrics = document.getElementById("lyrics").value
        genre = document.getElementsByClassName("genre")
        for (let i = 0; i < genre.length; i++) {
            cur_genre = genre[i].id;
            if (document.getElementById(cur_genre).className == badge_checked) {
                select_genre = document.getElementById(cur_genre).textContent
            }
        }
        const albumInput = {
            song_names: select_song,
            artist_name: select_artist,
            genre: select_genre,
            album_name: select_album,
            lyric: select_lyrics,
        };

        if (document.getElementById("listGroupRadioGrid1").checked == true) {
            select_model = document.getElementById("listGroupRadioGrid1").value
        }
        else {
            select_model = document.getElementById("listGroupRadioGrid2").value
        }
        if (select_model == "Stable Diffusion") {
            // 이미지 생성시간 안내 모달창 띄우기
            $('#create_modal1').modal('show');

            // 스피너 보이기
            document.getElementById("create_spinner").style.display = "block"

            try {
                const response = await fetch('http://localhost:8000/generate_cover', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(albumInput),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                console.log(data.images);  // Log the images data to check if it's correct
                for (let i = 1; i <= 4; i++) {
                    let imgElement = document.getElementById(`image${i}`);
                    // Log img element
                    // console.log(imgElement);
                    // console.log(data.images[i-1]);
                    imgElement.src = 'data:image/jpeg;base64,' + data.images[i - 1];
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }
        else {  // DreamBooth
            const imagePreview = document.querySelector('#imagePreview');
            const images = imagePreview.querySelectorAll('img');
            const imageUrls = Array.from(images).map(img => img.src);
            console.log(imageUrls.length)
            console.log('업로드한 이미지 목록들:', imageUrls);
            if (imageUrls.length == 0) {
                alert('이미지를 업로드해주세요.')
                document.getElementById("imageUpload").click();
                return;
            }

            const genderButtons = document.querySelectorAll('.gender');
            genderButtons.forEach((button) => {
                if (button.checked) {
                    const selectedGender = button.value;
                    console.log(selectedGender)   // 성별: 'man', 'woman'
                }
            });

            // 이미지 생성시간 안내 모달창 띄우기
            $('#create_modal2').modal('show');

            // TODO: DreamBooth 백엔드 연결 구현
        }


        document.getElementById("create_spinner").style.display = "none";
        document.getElementById("info_alert").style.display = "block";
        watermark = document.getElementsByClassName("watermark");
        for (let i = 0; i < watermark.length; i++) {
            watermark[i].style.display = "block";
        }
        download_btn = document.getElementsByClassName("download_btn");
        for (let i = 0; i < download_btn.length; i++) {
            download_btn[i].style.pointerEvents = "auto";
        }
    })

    document.querySelectorAll(".genre").forEach(obj => {
        obj.addEventListener("click", () => {
            document.querySelectorAll(".genre").forEach(obj2 => {
                document.getElementById(obj2.id).className = badge_not_checked
            })
            document.getElementById(obj.id).className = badge_checked
        })
    })
})
