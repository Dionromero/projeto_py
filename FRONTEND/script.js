const btn = document.getElementById("btnConsultar");
const texto = document.getElementById("resultadoTexto");

// Função para carregar a tabela climática
async function carregarClima() {
    try {
        const response = await fetch("http://127.0.0.1:5000/api/clima"); // sua rota do backend
        if (!response.ok) throw new Error("Erro ao buscar dados do clima");

        const dados = await response.json();
        const tabela = document.getElementById("tabelaClima").getElementsByTagName("tbody")[0];

        // Limpa tabela
        tabela.innerHTML = "";

        // Filtra horários de 5 em 5 horas
        dados.forEach(item => {
            const hora = new Date(item.timestamp).getHours();
            if (hora % 5 === 0) {
                const row = tabela.insertRow();
                row.insertCell(0).innerText = `${hora}:00`;
                row.insertCell(1).innerText = item.temperatura.toFixed(1);
                row.insertCell(2).innerText = item.umidade;
                row.insertCell(3).innerText = item.condicao;
            }
        });

    } catch (erro) {
        console.error(erro);
        const tabela = document.getElementById("tabelaClima").getElementsByTagName("tbody")[0];
        tabela.innerHTML = `<tr><td colspan="4">❌ Erro ao carregar dados do clima</td></tr>`;
    }
}

// Evento do botão de recomendação
btn.addEventListener("click", async () => {
    texto.innerText = "⏳ Consultando modelo...";

    try {
        const resp = await fetch("http://127.0.0.1:5000/recomendar");
        const data = await resp.json();

        texto.innerText = `👕 Recomendação da IA: classe ${data.recomendacao}`;

        // Depois de mostrar a recomendação, carrega a tabela climática
        await carregarClima();

    } catch (erro) {
        texto.innerText = "❌ Erro ao consultar o servidor. Verifique se o backend está rodando.";
        console.error(erro);
    }
});

// Atualiza a tabela climática a cada 5 minutos automaticamente
setInterval(carregarClima, 5 * 60 * 1000);
