const btn = document.getElementById("btnConsultar");
const generoSelect = document.getElementById("generoSelect");
const infoClimaTexto = document.getElementById("infoClimaTexto");

const elCabeca = document.getElementById("itemCabeca");
const elTronco = document.getElementById("itemTronco");
const elPernas = document.getElementById("itemPernas");
const elPes = document.getElementById("itemPes");

const tempDestaque = document.getElementById("tempDestaque");
const condicaoDestaque = document.getElementById("condicaoDestaque");
const detalheUmid = document.getElementById("detalheUmid");
const detalheChuvaHero = document.getElementById("detalheChuvaHero");
const dataHoje = document.getElementById("dataHoje");

function obterIconeClima(condicao) {
    if (!condicao) return "☁️";
    const c = condicao.toLowerCase();
    if (c.includes("sol") || c.includes("limpo")) return "☀️";
    if (c.includes("chuva") || c.includes("garoa")) return "🌧️";
    if (c.includes("trovoada") || c.includes("raio")) return "⛈️";
    if (c.includes("nublado") || c.includes("nuvens")) return "☁️";
    if (c.includes("parcial")) return "⛅";
    if (c.includes("neblina")) return "🌫️";
    return "🌥️"; 
}

async function carregarClima() {
    try {
        const response = await fetch("http://127.0.0.1:5000/api/clima"); 
        if (!response.ok) throw new Error("Erro API Clima");
        const dados = await response.json();
        
        const hoje = new Date();
        if(dataHoje) dataHoje.innerText = hoje.toLocaleDateString('pt-BR', { weekday: 'short', day: 'numeric' });

        if(dados.length > 0) {
            // Pega o horário atual (mais próximo)
            const horaAtual = new Date().getHours();
            // Tenta achar o dado da hora atual na lista, senão pega o primeiro
            const atual = dados.find(d => new Date(d.timestamp).getHours() === horaAtual) || dados[0];

            if(tempDestaque) tempDestaque.innerText = Math.round(atual.temperatura);
            if(condicaoDestaque) condicaoDestaque.innerText = atual.condicao;
            if(detalheUmid) detalheUmid.innerText = `💧 ${atual.umidade}%`;
            if(detalheChuvaHero) detalheChuvaHero.innerText = `☔ ${atual.chance_of_rain}%`;
        }

        const tabela = document.getElementById("tabelaClima").getElementsByTagName("tbody")[0];
        tabela.innerHTML = "";
        
        dados.forEach((item, index) => {
            const horaNum = new Date(item.timestamp).getHours();
            
            // Mostra a cada 2 horas para caber na tela
            if (index < 24 && horaNum % 2 === 0) { 
                const row = tabela.insertRow();
                row.className = "hover:bg-white/10 transition duration-200 group border-b border-white/5";

                // Hora
                const cHora = row.insertCell(0);
                cHora.className = "py-3 px-1 text-slate-400 font-mono text-[10px]";
                cHora.innerText = `${String(horaNum).padStart(2, '0')}:00`;

                // Ícone
                const cIcon = row.insertCell(1);
                cIcon.className = "py-3 px-1 text-center text-base";
                cIcon.innerText = obterIconeClima(item.condicao);

                // Temp
                const cTemp = row.insertCell(2);
                cTemp.className = "py-3 px-1 font-bold text-white text-xs";
                cTemp.innerText = `${Math.round(item.temperatura)}°`;

                // Chuva (Ajustado)
                const cChuva = row.insertCell(3);
                // Garante que é número
                const probChuva = Number(item.chance_of_rain);
                
                cChuva.className = "py-3 px-1 text-center text-xs";
                
                if (probChuva > 0) {
                    // Se for maior que 0, mostra azul
                    cChuva.innerHTML = `<span class="text-blue-400 font-bold">${probChuva}%</span>`;
                } else {
                    // Se for 0 cravado
                    cChuva.innerHTML = `<span class="text-slate-600 opacity-40">0%</span>`;
                }

                // Vento
                const cVento = row.insertCell(4);
                const velVento = Math.round(item.vento_kph || 0);
                cVento.className = "py-3 px-1 text-right text-xs text-slate-400";
                cVento.innerText = `${velVento} km`;
            }
        });

    } catch (e) { 
        console.error(e);
        if(condicaoDestaque) condicaoDestaque.innerText = "Offline";
    }
}

btn.addEventListener("click", async () => {
    const genero = generoSelect.value;
    const originalText = btn.innerHTML;
    
    btn.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Gerando...`;
    btn.disabled = true;

    try {
        const resp = await fetch(`http://127.0.0.1:5000/recomendar?genero=${genero}&local=Curitiba`);
        const data = await resp.json();

        const updateField = (el, text) => {
            el.style.opacity = 0;
            setTimeout(() => {
                el.innerText = text;
                el.style.opacity = 1;
            }, 200);
        };

        updateField(elCabeca, data.look.cabeca);
        
        let troncoTexto = data.look.tronco;
        if (data.look.casaco) troncoTexto = `${data.look.casaco} + ${data.look.tronco}`;
        updateField(elTronco, troncoTexto);

        updateField(elPernas, data.look.pernas);
        updateField(elPes, data.look.pes);

        infoClimaTexto.innerText = `Look ideal para ${data.temperatura}°C (${data.categoria_clima})`;
        
        await carregarClima();

    } catch (err) {
        console.error(err);
        alert("Erro ao conectar.");
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

window.onload = carregarClima;