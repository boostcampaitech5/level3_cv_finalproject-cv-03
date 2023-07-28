async function fetchAlbumImages() {
    try {
        const response = await fetch('http://49.50.167.24:30008/api/get_album_images', { method: 'GET',

        mode: "cors",
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json'
            // 'Content-Type': 'application/x-www-form-urlencoded',
          },});

        const data = await response.json();

        if (response.ok) {
            document.getElementById('example-images').style.display = "block"
            for (let i = 0; i < data.album_images.length; i++) {
                const albumImage = data.album_images[i];
                document.getElementById('album-image-'+(i+1)).src = albumImage.url;
                document.getElementById('album-prompt-'+(i+1)).innerHTML =
                    `<strong>Song:</strong> ${albumImage.song_names}</br>
                    <strong>Artist:</strong> ${albumImage.artist_name}<br/>
                    <strong>Album:</strong> ${albumImage.album_name}</br>
                    <strong>Genre:</strong> ${albumImage.genre}`;
            };
        } else {
            console.error('Error fetching album images:', data.message);
        }
    } catch (error) {
        console.error('Error fetching album images:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetchAlbumImages();
});
