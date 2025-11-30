// home.js
const btn = document.getElementById("btnConsultar");
const texto = document.getElementById("resultadoTexto");
const imagemRoupa = document.getElementById("imagemRoupa");

// --- Função para buscar e exibir o clima ---
async function carregarClima() {
    try {
        const response = await fetch("http://127.0.0.1:5000/api/clima"); 
        if (!response.ok) throw new Error("Erro ao buscar dados do clima");

        const dados = await response.json();
        const tabela = document.getElementById("tabelaClima").getElementsByTagName("tbody")[0];
        tabela.innerHTML = "";

        // Preenche a tabela
        dados.forEach(item => {
            const hora = new Date(item.timestamp).getHours();
            // Mostra a cada 4 horas para não lotar a tabela
            if (hora % 4 === 0) {
                const row = tabela.insertRow();
                row.insertCell(0).innerText = `${String(hora).padStart(2, '0')}:00`;
                row.insertCell(1).innerText = `${item.temperatura.toFixed(1)}°C`;
                row.insertCell(2).innerText = `${item.umidade}%`;
                row.insertCell(3).innerText = item.condicao;
            }
        });
    } catch (erro) {
        console.error("Erro clima:", erro);
        const tabela = document.getElementById("tabelaClima").getElementsByTagName("tbody")[0];
        tabela.innerHTML = `<tr><td colspan="4">Não foi possível carregar o clima.</td></tr>`;
    }
}

// --- Função Principal: Recomendação ---
btn.addEventListener("click", async () => {
    texto.innerText = "Consultando inteligência artificial...";
    if(imagemRoupa) imagemRoupa.style.display = "none";

    try {
        const resp = await fetch("http://127.0.0.1:5000/recomendar");
        const data = await resp.json();

        // 1. Atualiza o texto
        texto.innerHTML = `Recomendação: <strong>${data.nome_roupa}</strong>`;

        // 2. Atualiza a IMAGEM
        if (data.imagem_path && imagemRoupa) {
            const BASE_URL = "http://127.0.0.1:5000/";
            
            // Tratamento da URL da imagem
            if (data.imagem_path.startsWith('http')) {
                imagemRoupa.src = data.imagem_path;
            } else {
                const caminhoLimpo = data.imagem_path.startsWith('/') ? data.imagem_path.substring(1) : data.imagem_path;
                imagemRoupa.src = BASE_URL + caminhoLimpo;
            }
            
            imagemRoupa.style.display = "block";
        } else {
            texto.innerHTML += "<br><small>(Imagem de exemplo não disponível)</small>";
        }

        // Atualiza o clima também ao clicar
        await carregarClima();

    } catch (err) {
        console.error(err);
        texto.innerText = "Erro ao conectar com o servidor.";
    }
});

// Carrega o clima ao abrir a página
window.onload = carregarClima;