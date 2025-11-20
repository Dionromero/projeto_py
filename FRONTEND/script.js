const btn = document.getElementById("btnConsultar");
const texto = document.getElementById("resultadoTexto");
const imagemRoupa = document.getElementById("imagemRoupa"); // Captura a tag img

// --- Função para buscar e exibir o clima ---
async function carregarClima() {
    try {
        // Conecta ao backend na porta 5000
        const response = await fetch("http://127.0.0.1:5000/api/clima"); 
        if (!response.ok) throw new Error("Erro ao buscar dados do clima");

        const dados = await response.json();
        const tabela = document.getElementById("tabelaClima").getElementsByTagName("tbody")[0];
        tabela.innerHTML = "";

        // Preenche a tabela
        dados.forEach(item => {
            const hora = new Date(item.timestamp).getHours();
            // Filtra para mostrar a cada 5 horas (ex: 00:00, 05:00...)
            if (hora % 5 === 0) {
                const row = tabela.insertRow();
                row.insertCell(0).innerText = `${String(hora).padStart(2, '0')}:00`;
                row.insertCell(1).innerText = item.temperatura.toFixed(1);
                row.insertCell(2).innerText = item.umidade;
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
    
    // Esconde a imagem enquanto carrega
    if(imagemRoupa) imagemRoupa.style.display = "none";

    try {
        // Chama a recomendação
        const resp = await fetch("http://127.0.0.1:5000/recomendar");
        const data = await resp.json();

        // 1. Atualiza o texto com o NOME da roupa
        texto.innerHTML = `Recomendação: <strong>${data.nome_roupa}</strong> (Classe ${data.recomendacao})`;

        // 2. Atualiza a IMAGEM se o caminho existir
        if (data.imagem_path && imagemRoupa) {
            imagemRoupa.src = data.imagem_path; // Define a URL da imagem
            imagemRoupa.style.display = "block"; // Torna visível
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