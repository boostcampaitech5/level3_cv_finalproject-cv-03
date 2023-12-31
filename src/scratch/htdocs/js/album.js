// 페이지 로드 시 로그인 상태 확인
window.onload = function () {
    updateLoginState();
};
// 임시 서버주소
const server_domain = 'http://34.22.72.143:80'
async function LoginInfo(user) {
    try {
        const response = await fetch(server_domain + '/api/user', {
            method: 'POST',
            mode: "cors",
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(user),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log(data);
        sessionStorage.setItem('user_id', data.user_id);
    } catch (error) {
        console.error('Error:', error);
    }
}

//카카오 로그인
function kakaoLogin() {
    Kakao.Auth.login({
        success: function (response) {
            Kakao.API.request({
                url: '/v2/user/me',
                success: function (response) {
                    const user = {
                        nickname: response.kakao_account.profile.nickname,
                        age_range: response.kakao_account.age_range,
                        email: response.kakao_account.email
                    }
                    sessionStorage.setItem('isLoggedIn', 'true');
                    sessionStorage.setItem('user_nickname', user.nickname);
                    LoginInfo(user);
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
                sessionStorage.removeItem('user_id');
                sessionStorage.removeItem('user_nickname');
                updateLoginState();
                window.location.href = 'index.html';
                window.onload()
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
        document.getElementById('kakao_logout').style.display = 'block';
        document.getElementById('text_login').innerText = sessionStorage.getItem('user_nickname') + '님'
    } else {
        // 로그아웃 상태
        document.getElementById('kakao_login').style.display = 'block';
        document.getElementById('kakao_logout').style.display = 'none';
        document.getElementById('text_login').innerText = '';
    }
}

// 경과 시간을 갱신하는 함수
function updateElapsedTime(startTimestamp) {
    const now = new Date();
    const elapsedSeconds = Math.floor((now - startTimestamp) / 1000);
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = elapsedSeconds % 60;
    document.getElementById('elapsed-time').textContent = `경과 시간: ${minutes}분 ${seconds}초`;
}

function imageDownload(num) {
    console.log('image download ', num)
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

const badge_checked = "badge bg-primary-subtle border border-primary-subtle text-primary-emphasis rounded-pill mb-1 genre"
const badge_not_checked = "badge bg-secondary-subtle border border-secondary-subtle text-secondary-emphasis rounded-pill mb-1 genre"
function resetInput() {
    document.getElementById("song_name").value = ''
    document.getElementById("artist_name").value = ''
    document.getElementById("album_name").value = ''
    document.getElementById("lyrics").value = ''
    genre = document.getElementsByClassName("genre")
    for (let i = 0; i < genre.length; i++) {
        genre[i].className = badge_not_checked
    }
    const previewContainer = document.querySelector('#imagePreview');
    while (previewContainer.firstChild) {
        previewContainer.removeChild(previewContainer.firstChild);
    }
    document.getElementById("imageUpload").value = null
    document.querySelectorAll('.gender')[0].checked = true
    document.querySelectorAll('.gender')[1].checked = false
    document.getElementById('elapsed-time').textContent = '경과 시간: 0분 0초'
    image = document.getElementsByClassName("created_image")
    for (let i = 0; i < image.length; i++) {
        image[i].src = "images/empty.jpg"
    }
    watermark = document.getElementsByClassName("watermark");
    for (let i = 0; i < watermark.length; i++) {
        watermark[i].style.display = "none";
    }
    download_btn = document.getElementsByClassName("download_btn");
    for (let i = 0; i < download_btn.length; i++) {
        download_btn[i].style.pointerEvents = "none";
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.querySelector('#mypage_login').addEventListener('click', (event) => {
        event.preventDefault();
        user_id = sessionStorage.getItem('user_id');
        if (user_id == null) {
            alert("로그인 후 사용가능합니다.");
        }
        else {
            window.location.href = 'mypage.html';
        }
    })


    select_model = document.querySelectorAll('.select_model')
    model2_contents = document.querySelectorAll('.model_content')
    select_model.forEach((radio) => {
        radio.addEventListener('change', () => {
            if (radio.checked) {
                resetInput()
                if (radio.id == "listGroupRadioGrid2") {
                    document.getElementById('model2_content').style.display = "block"
                }
                else {
                    document.getElementById('model2_content').style.display = "none"
                }
            }
        })
    })

    image_upload = document.querySelector('#imageUpload');
    if (image_upload) {
        image_upload.addEventListener('change', function () {
            const selectedFiles = this.files;
            const previewContainer = document.querySelector('#imagePreview');

            // 이미지 갯수가 최소 4장, 최대 10장인지 확인
            if (selectedFiles.length < 4 || selectedFiles.length > 10) {
                alert('최소 4장에서 최대 10장의 이미지를 업로드해주세요.');
                document.getElementById("imageUpload").value = null
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
    }


    // 모달을 열 때 클릭된 버튼 숫자정보(1~4) 가져오기
    // bring the number of the clicked button of the modal

    let imageUrl;
    let buttonType;
    $('#download_modal').on('show.bs.modal', function (event) {
        let button = $(event.relatedTarget);
        buttonType = button.data('button-type');
        console.log(buttonType); // check the value of buttonType
        console.log(typeof buttonType)
        // Determine the id of the image to be reviewed
        let imageId = 'image' + buttonType;

        // Get the image element
        let imageElement = document.getElementById(imageId);

        // Get the image url
        console.log(imageElement.src);
        imageUrl = imageElement.src;
    });

    review_send_btn = document.querySelector("#review_send_btn");
    if (review_send_btn) {
        review_send_btn.addEventListener("click", async (e) => {
            e.preventDefault()
            user_starpoint = 0
            user_review = document.getElementById("review_comment").value
            for (let i = 1; i <= 10; i++) {
                cur_review = "starpoint_" + i
                if (document.getElementById(cur_review).checked == true) {
                    user_starpoint = (i / 2);
                    break;
                }
            }
            if (user_starpoint == 0 || user_review == "") {
                alert('별점과 리뷰를 모두 작성해주세요.');
                document.getElementById("review_comment").focus();
            }
            else {
                console.log('review login user id:', sessionStorage.getItem('user_id'));
                user_id = sessionStorage.getItem('user_id')

                const UserReviewInput = {
                    output_id: output_id,
                    url_id: buttonType,
                    user_id: sessionStorage.getItem('user_id') !== null ? sessionStorage.getItem('user_id') : '',
                    rating: user_starpoint,
                    comment: user_review
                }

            try {
                console.log("Review data being sent:", UserReviewInput);
                const response = await fetch(server_domain + '/api/review', {
                    method: 'POST',
                    mode: "cors",
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(UserReviewInput),
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
}

    let select_artist = '';
    let select_song = '';
    let select_genre = '';
    let select_album = '';

    let output_id;
    img_create_btn = document.querySelector("#img_create_btn");
    if (img_create_btn) {
        img_create_btn.addEventListener("click", async (e) => {
            e.preventDefault()
            // 버튼 동작 체크
            console.log("Button Clicked!");
            const startTimestamp = Date.now();

            const required_ids = ["song_name", "artist_name"];
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
            select_lyric = document.getElementById("lyrics").value
            select_genre = ''
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
                lyric: select_lyric,
            };
            if (document.getElementById("listGroupRadioGrid1").checked == true) {
                select_model = document.getElementById("listGroupRadioGrid1").value
            }
            else {
                select_model = document.getElementById("listGroupRadioGrid2").value
            }

            UserAlbumInput = {
                user_id: sessionStorage.getItem('user_id') !== null ? sessionStorage.getItem('user_id') : '',
                model: select_model,
                song_name: select_song,
                artist_name: select_artist,
                album_name: select_album,
                genre: select_genre,
                lyric: select_lyric,
                gender: '',
                image_urls: [],
            };

            if (select_model == "Stable Diffusion") {
                // 이미지 생성시간 안내 모달창 띄우기
                $('#create_modal1').modal('show');

                // 스피너 보이기
                document.getElementById("create_spinner").style.display = "block"
                // 1초마다 경과 시간을 갱신
                timerId = setInterval(() => {
                    updateElapsedTime(startTimestamp);
                }, 1000);

                try {
                    await generateCover(UserAlbumInput);
                } catch (error) {
                    console.error('Error:', error);
                }

                async function generateCover(UserAlbumInput) {
                    const response = await fetch(server_domain + '/api/generate_cover', {
                        method: 'POST',
                        mode: "cors",
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(UserAlbumInput),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        checkTaskStatus(data.task_id);
                    } else {
                        console.error(`HTTP error! status: ${response.status}`);
                    }
                }

                async function checkTaskStatus(taskId) {
                    const response = await fetch(`${server_domain}/api/get_task_result/${taskId}`, {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.status === 'SUCCESS') {
                            getTaskResult(taskId);

                            clearInterval(timerId);
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

                        } else {
                            setTimeout(() => checkTaskStatus(taskId), 1000);
                        }
                    } else {
                        console.error(`HTTP error! status: ${response.status}`);
                    }
                }
                async function getTaskResult(taskId) {
                    const response = await fetch(`${server_domain}/api/get_task_result/${taskId}`, {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    });

                    if (response.ok) {
                        const data = await response.json();
                        output_id = data.result.output_id;
                        console.log(data)
                        console.log(data.result.output_id)
                        console.log(data.result)
                        for (let i = 1; i <= 4; i++) {
                            let imgElement = document.getElementById(`image${i}`);
                            imgElement.src = data.result.image_urls[i - 1];
                        }
                    } else {
                        console.error(`HTTP error! status: ${response.status}`);
                    }
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
                let selectedGender = '';
                genderButtons.forEach((button) => {
                    if (button.checked) {
                        selectedGender = button.value;
                        console.log(selectedGender)   // 성별: 'man', 'woman'
                    }
                });
                UserAlbumInput.gender = selectedGender
                UserAlbumInput.image_urls = imageUrls // TODO: 스토리지 주소로 바꿔야할까?

                // 이미지 생성시간 안내 모달창 띄우기
                $('#create_modal2').modal('show');
                document.getElementById("create_spinner").style.display = "block"
                // 1초마다 경과 시간을 갱신
                timerId = setInterval(() => {
                    updateElapsedTime(startTimestamp);
                }, 1000);

                // TODO: DreamBooth back-end connect
                // Uploading each image
                for (let i = 0; i < imageUrls.length; i++) {
                    const imageUrl = imageUrls[i];

                    try {
                        // Fetch the image data from the URL
                        const response = await fetch(imageUrl);
                        const imageBlob = await response.blob();

                        // Create a FormData object and append the image blob to it
                        const formData = new FormData();
                        formData.append('image', imageBlob, `image${i}.jpg`);

                        // Send the image data to the server
                        const uploadResponse = await fetch(server_domain + '/api/upload_image', {
                            method: 'POST',
                            body: formData
                        });

                        if (!uploadResponse.ok) {
                            throw new Error(`HTTP error! status: ${uploadResponse.status}`);
                        }

                        // Parse the JSON response
                        const data = await uploadResponse.json();
                        console.log(data);
                    } catch (error) {
                        console.error('Error:', error);
                    }
                }
                try {
                    await train_inference(UserAlbumInput);
                } catch (error) {
                    console.error('Error:', error);
                }

                async function train_inference(UserAlbumInput) {
                    const user = { gender: selectedGender };
                    const response = await fetch(server_domain + '/api/train_inference', {
                        method: 'POST',
                        mode: "cors",
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(UserAlbumInput),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        checkTaskStatus_train(data.task_id);
                    } else {
                        console.error(`HTTP error! status: ${response.status}`);
                    }
                }

                async function checkTaskStatus_train(taskId) {
                    const response = await fetch(`${server_domain}/api/get_task_result/${taskId}`, {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.status === 'SUCCESS') {
                            getTaskResult(taskId);

                            clearInterval(timerId);
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

                        } else {
                            setTimeout(() => checkTaskStatus_train(taskId), 30000);
                        }
                    } else {
                        console.error(`HTTP error! status: ${response.status}`);
                    }
                }
                async function getTaskResult(taskId) {
                    const response = await fetch(`${server_domain}/api/get_task_result/${taskId}`, {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    });

                    if (response.ok) {
                        const data = await response.json();
                        output_id = data.result.output_id;
                        console.log(data)
                        console.log(data.result.output_id)
                        console.log(data.result)
                        for (let i = 1; i <= 4; i++) {
                            let imgElement = document.getElementById(`image${i}`);
                            imgElement.src = data.result.image_urls[i - 1];
                        }
                    } else {
                        console.error(`HTTP error! status: ${response.status}`);
                    }
                }
            }
        })
    }

    document.querySelectorAll(".genre").forEach(obj => {
        obj.addEventListener("click", () => {
            document.querySelectorAll(".genre").forEach(obj2 => {
                document.getElementById(obj2.id).className = badge_not_checked
            })
            document.getElementById(obj.id).className = badge_checked
        })
    })
})
