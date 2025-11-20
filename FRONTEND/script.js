const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const btnEnviar = document.getElementById("btnEnviar");
const resultado = document.getElementById("resultado");

let imagemBase64 = null;

// Mostrar preview ao selecionar arquivo
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
        imagemBase64 = reader.result.split(",")[1]; // remove prefixo data:image/png...
        preview.src = reader.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);
});

// Enviar para backend
btnEnviar.addEventListener("click", async () => {
    if (!imagemBase64) {
        alert("Escolha uma imagem primeiro!");
        return;
    }

    resultado.innerText = "🔮 Processando...";

    const resp = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: imagemBase64 })
    });

    const data = await resp.json();

    if (data.error) {
        resultado.innerText = "❌ Erro: " + data.error;
    } else {
        resultado.innerText = "👕 Classe prevista: " + data.classe;
    }
});
