const btn = document.getElementById("btnConsultar");
const generoSelect = document.getElementById("generoSelect");
const estiloSelect = document.getElementById("estiloSelect");
const infoClimaTexto = document.getElementById("infoClimaTexto");

// Elementos da Toggle
const btnToggle = document.getElementById("btnToggleClima");
const containerTabela = document.getElementById("containerTabela");
const setaToggle = document.getElementById("setaToggle");

// Lógica de Abrir/Fechar Tabela
if(btnToggle && containerTabela) {
    btnToggle.addEventListener("click", () => {
        // Verifica se está aberto
        const isClosed = containerTabela.classList.contains("max-h-0");

        if (isClosed) {
            // ABRIR
            containerTabela.classList.remove("max-h-0", "opacity-0");
            containerTabela.classList.add("max-h-[400px]", "opacity-100"); // Altura máx suficiente
            setaToggle.classList.add("rotate-180"); // Gira a seta
        } else {
            // FECHAR
            containerTabela.classList.remove("max-h-[400px]", "opacity-100");
            containerTabela.classList.add("max-h-0", "opacity-0");
            setaToggle.classList.remove("rotate-180"); // Volta a seta
        }
    });
}

const parts = {
    cima:   { txt: document.getElementById("itemCima"),   img: document.getElementById("imgCima") },
    baixo:  { txt: document.getElementById("itemBaixo"),  img: document.getElementById("imgBaixo") },
    casaco: { txt: document.getElementById("itemCasaco"), img: document.getElementById("imgCasaco") }
};

// --- ALTERAÇÃO PRINCIPAL AQUI ---
// Apontando para o seu servidor no Render
const BASE_URL = "https://wardobre-ipynb.onrender.com"; 

function obterIconeClima(textoCondicao) {
    if (!textoCondicao) return "☁️";
    const c = textoCondicao.toLowerCase();
    if (c.includes("sol") || c.includes("limpo") || c.includes("clear")) return "☀️";
    if (c.includes("chuva") || c.includes("rain") || c.includes("drizzle")) return "🌧️";
    if (c.includes("trovoada") || c.includes("thunder")) return "⛈️";
    if (c.includes("nublado") || c.includes("cloud")) return "☁️";
    return "🌥️"; 
}

async function carregarClima() {
    try {
        const response = await fetch(`${BASE_URL}/api/clima`); 
        if (!response.ok) return;
        const dados = await response.json();
        
        // Data
        const hoje = new Date();
        const opcoesData = { day: 'numeric', month: 'short' };
        const elData = document.getElementById("dataHoje");
        if(elData) elData.innerText = hoje.toLocaleDateString('pt-BR', opcoesData).replace('.', '');

        // Card Hero
        if(dados && dados.current) {
            document.getElementById("tempDestaque").innerText = Math.round(dados.current.temp_c);
            document.getElementById("condicaoDestaque").innerText = dados.current.condition.text;
            document.getElementById("detalheUmid").innerText = `${dados.current.humidity}%`;
            
            // Ícone Grande
            const iconHero = obterIconeClima(dados.current.condition.text);
            const iconEl = document.getElementById("iconDestaque");
            if(iconEl) iconEl.innerText = iconHero;

            let chuva = 0;
            if(dados.forecast) chuva = dados.forecast.forecastday[0].hour[0].chance_of_rain;
            document.getElementById("detalheChuvaHero").innerText = `${chuva}%`;
        }

        // Tabela
        const tabelaEl = document.getElementById("tabelaClima");
        if(tabelaEl && dados.forecast) {
            const tabelaBody = tabelaEl.getElementsByTagName("tbody")[0];
            tabelaBody.innerHTML = "";
            dados.forecast.forecastday[0].hour.forEach((item, index) => {
                if (index % 3 === 0) { // A cada 3h para ficar mais compacto
                    const row = tabelaBody.insertRow();
                    row.className = "hover:bg-white/5 transition-colors border-b border-slate-800/50";
                    
                    const hora = item.time.split(' ')[1];
                    const icon = obterIconeClima(item.condition.text);
                    
                    row.innerHTML = `
                        <td class="py-3 px-2 font-mono text-slate-400">${hora}</td>
                        <td class="py-3 px-2 text-center text-base">${icon}</td>
                        <td class="py-3 px-2 font-bold text-white text-right">${Math.round(item.temp_c)}°</td>
                        <td class="py-3 px-2 text-center text-xs text-blue-400 font-medium">${item.chance_of_rain}%</td>
                    `;
                }
            });
        }
    } catch (e) { console.warn("Clima indisponível ou servidor acordando..."); }
}

btn.addEventListener("click", async () => {
    const genero = generoSelect.value;
    const estilo = estiloSelect.value;
    const originalText = btn.innerHTML;
    
    // Animação de Loading
    btn.innerHTML = `
        <svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg> Gerando...`;
    btn.disabled = true;

    try {
        const resp = await fetch(`${BASE_URL}/recomendar?genero=${genero}&local=Curitiba&estilo=${estilo}`);
        const data = await resp.json();

        // Função auxiliar para atualizar UI
        const updatePart = (partKey, textVal, imgUrl) => {
            const el = parts[partKey];
            if(!el) return;
            el.txt.style.opacity = 0;
            if(el.img) el.img.style.opacity = 0;

            setTimeout(() => {
                if (!textVal || textVal === "-" || textVal === "Peça Única" || textVal === "Nada") {
                    el.txt.innerText = (textVal === "Peça Única") ? "(Peça Única)" : "—";
                    if(el.img) el.img.src = "https://placehold.co/1x1/ffffff/ffffff"; // Imagem em branco/transparente
                } else {
                    el.txt.innerText = textVal;
                    if (imgUrl && el.img) {
                        // Verifica se a URL já é completa ou precisa concatenar
                        let src = imgUrl.startsWith('http') || imgUrl.startsWith('data') ? imgUrl : BASE_URL + imgUrl;
                        el.img.src = src;
                        el.img.onload = () => { el.img.style.opacity = 1; };
                    }
                }
                el.txt.style.opacity = 1;
            }, 200);
        };

        // --- CORREÇÃO DE ESTRUTURA DE DADOS ---
        // Se o backend retornar direto data.cima (como no seu Python), usa ele.
        // Se retornar data.look.cima (estrutura antiga), usa o fallback.
        const cimaVal = data.cima || (data.look ? data.look.cima : "-");
        const baixoVal = data.baixo || (data.look ? data.look.baixo : "-");
        const casacoVal = data.casaco || (data.look ? data.look.casaco : "-");

        updatePart("cima", cimaVal, data.imagens?.cima);
        updatePart("baixo", baixoVal, data.imagens?.baixo);
        updatePart("casaco", casacoVal, data.imagens?.casaco);

        if(infoClimaTexto) {
            infoClimaTexto.innerText = `Look ideal para ${data.temperatura}°C (${data.categoria_clima})`;
        }
        
        await carregarClima();

    } catch (err) { 
        console.error(err);
        alert("O servidor pode estar 'dormindo'. Aguarde 30s e tente de novo!"); 
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

window.onload = carregarClima;