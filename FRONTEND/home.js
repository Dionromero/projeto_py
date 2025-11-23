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
            // Mostra a cada 5 horas ou ajusta conforme necessidade
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
        tabela.innerHTML = `<tr><td colspan="4">❌ Erro de conexão com o Backend</td></tr>`;
    }
}

// --- Evento do Botão de Recomendação ---
btn.addEventListener("click", async () => {
    texto.innerText = "⏳ Consultando IA...";
    if(imagemRoupa) imagemRoupa.style.display = "none";

    try {
        const resp = await fetch("http://127.0.0.1:5000/recomendar");
        const data = await resp.json();

        // 1. Atualiza o texto
        texto.innerHTML = `Recomendação: <strong>${data.nome_roupa}</strong>`;

        // 2. Atualiza a IMAGEM (Correção do caminho)
        if (data.imagem_path && imagemRoupa) {
            const BASE_URL = "http://127.0.0.1:5000/";
            
            // Se já for um link completo (http...), usa ele. Se não, concatena.
            if (data.imagem_path.startsWith('http')) {
                imagemRoupa.src = data.imagem_path;
            } else {
                // Remove barra inicial se houver para evitar duplicidade
                const caminhoLimpo = data.imagem_path.startsWith('/') ? data.imagem_path.substring(1) : data.imagem_path;
                imagemRoupa.src = BASE_URL + caminhoLimpo;
            }
            
            imagemRoupa.style.display = "block";
        } else {
            texto.innerHTML += "<br><small>(Imagem de exemplo não disponível)</small>";
        }

        // Atualiza o clima também
        await carregarClima();

    } catch (erro) {
        texto.innerText = "❌ Erro ao consultar servidor.";
        console.error("Erro recomendação:", erro);
    }
});

// Inicializa o clima ao abrir a página
carregarClima();
setInterval(carregarClima, 5 * 60 * 1000);