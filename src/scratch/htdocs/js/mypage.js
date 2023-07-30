async function fetchAlbumImages(user_id) {
    try {
        const response = await fetch('http://34.22.72.143:80/api/get_album_images?user='+user_id, { method: 'GET',

        mode: "cors",
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json'
            // 'Content-Type': 'application/x-www-form-urlencoded',
          },});

        const data = await response.json();

        if (response.ok) {
            for (let i = 0; i < data.album_images.length; i++) {
                const albumImage = data.album_images[i];
                const imgElement = document.createElement('img');
                imgElement.src = albumImage.url;
                imgElement.addEventListener('click', () => {
                    showImageModal(albumImage);
                });

                imageContainer.querySelector('.image-grid').appendChild(imgElement);
            };
        } else {
            console.error('Error fetching album images:', data.message);
        }
    } catch (error) {
        console.error('Error fetching album images:', error);
    }
}

// 이미지를 동적으로 추가하는 함수
function addImagesToContainer(userImages) {
    userImages.forEach((image) => {
        const imgElement = document.createElement('img');
        imgElement.src = image['url'];
        imgElement.alt = '이미지';

        imgElement.addEventListener('click', () => {
            showImageModal(image);
        });

        imageContainer.querySelector('.image-grid').appendChild(imgElement);
    });
}

function showImageModal(image) {
    const modalBody = document.querySelector('.modal-body');
    // 이미지 클릭 시 모달의 내용을 새로운 정보로 업데이트
    document.getElementById('create_date').innerText = image.create_date;
    document.getElementById('song_name').innerText = image.song_name;
    document.getElementById('artist_name').innerText = image.artist_name;
    document.getElementById('genre').innerText = image.genre;
    document.getElementById('album_name').innerText = image.album_name;
    document.getElementById('lyric').innerText = image.lyric;

    // 모달 띄우기
    const imageModal = new bootstrap.Modal(document.getElementById('imageModal'));
    imageModal.show();
  }

document.addEventListener("DOMContentLoaded", () => {
    const imageContainer = document.getElementById('imageContainer');
    user_id = sessionStorage.getItem('user_id')
    if (user_id != null) {
        document.getElementById('logincheck').innerText = "이미지를 클릭하면 생성시 입력했던 정보를 볼 수 있습니다."
        fetchAlbumImages(user_id)
    }
    else {
        document.getElementById('logincheck').innerText = "*로그인 후 사용가능합니다.*"
    }
})
