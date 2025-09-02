// map_loader.js
function loadKakaoMapScript() {
    const script = document.createElement('script');
    script.src = '//dapi.kakao.com/v2/maps/sdk.js?appkey=f0f9ee9bf1253f56d0360bb4739aaeab&libraries=services,clusterer';
    script.async = true;
    script.onload = () => {
        console.log('Kakao Maps SDK loaded.');
        window.dispatchEvent(new CustomEvent('mapReady'));
    };
    document.head.appendChild(script);
}

loadKakaoMapScript();