const btn = document.getElementById("btnConsultar");
const generoSelect = document.getElementById("generoSelect");
const infoClimaTexto = document.getElementById("infoClimaTexto");

// Elementos do Look
const elCabeca = document.getElementById("itemCabeca");
const elTronco = document.getElementById("itemTronco");
const elPernas = document.getElementById("itemPernas");
const elPes = document.getElementById("itemPes");

// Elementos da Sidebar
const tempDestaque = document.getElementById("tempDestaque");
const condicaoDestaque = document.getElementById("condicaoDestaque");
const hojeMaxMin = document.getElementById("hojeMaxMin");

async function carregarClima() {
    try {
        const response = await fetch("http://127.0.0.1:5000/api/clima"); 
        if (!response.ok) throw new Error("Erro API Clima");
        const dados = await response.json();
        
        // 1. Preenche o Destaque (Hero)
        if(dados.length > 0) {
            const atual = dados[0]; 
            if(tempDestaque) tempDestaque.innerText = Math.round(atual.temperatura);
            if(condicaoDestaque) condicaoDestaque.innerText = atual.condicao;
            
            // Simulação de Max/Min (Se a API mandar no futuro, ajuste aqui)
            if(hojeMaxMin) hojeMaxMin.innerText = `Umidade: ${atual.umidade}% | Vento: ${atual.vento_kph.toFixed(0)} km/h`;
        }

        // 2. Preenche a Tabela Completa
        const tabela = document.getElementById("tabelaClima").getElementsByTagName("tbody")[0];
        tabela.innerHTML = "";
        
        dados.forEach((item, index) => {
            const hora = new Date(item.timestamp).getHours();
            
            // Mostra a cada 3 horas para não poluir
            if (index < 16 && hora % 2 === 0) { 
                const row = tabela.insertRow();
                
                // Estilo das células (Tailwind classes aplicadas via JS ou herdadas da TR)
                row.className = "hover:bg-slate-800 transition duration-200 border-b border-slate-800/50 last:border-0";

                // Hora
                const c1 = row.insertCell(0);
                c1.className = "py-3 px-2 font-bold text-slate-300";
                c1.innerText = `${String(hora).padStart(2, '0')}:00`;

                // Temperatura
                const c2 = row.insertCell(1);
                c2.className = "py-3 px-2 font-medium text-white";
                c2.innerText = `${Math.round(item.temperatura)}°`;

                // Chuva (Se não tiver dados, assume 0)
                const c3 = row.insertCell(2);
                const chanceChuva = item.chance_of_rain || 0;
                c3.className = `py-3 px-2 text-xs ${chanceChuva > 30 ? 'text-blue-400 font-bold' : 'text-slate-500'}`;
                c3.innerText = `${chanceChuva}%`;

                // Umidade
                const c4 = row.insertCell(3);
                c4.className = "py-3 px-2 text-xs text-slate-400";
                c4.innerText = `${item.umidade}%`;

                // Vento (Adicionei no backend ou pego do JSON se disponivel)
                const c5 = row.insertCell(4);
                c5.className = "py-3 px-2 text-xs text-slate-400";
                const vento = item.vento_kph || item.wind_kph || 0; // Tenta pegar vento
                c5.innerText = `${Math.round(vento)}km`;
            }
        });

    } catch (e) { 
        console.error(e);
        if(condicaoDestaque) condicaoDestaque.innerText = "Offline";
    }
}

btn.addEventListener("click", async () => {
    const genero = generoSelect.value;
    const textoOriginal = btn.innerHTML;
    
    // Animação simples no botão
    btn.innerHTML = `<svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Gerando...`;
    btn.disabled = true;

    try {
        const resp = await fetch(`http://127.0.0.1:5000/recomendar?genero=${genero}&local=Curitiba`);
        const data = await resp.json();

        // Atualiza UI com fade in
        const atualizar = (el, texto) => {
            el.style.opacity = 0;
            setTimeout(() => {
                el.innerText = texto;
                el.style.opacity = 1;
            }, 200);
        };

        atualizar(elCabeca, data.look.cabeca);
        
        let troncoTexto = data.look.tronco;
        if (data.look.casaco) troncoTexto = `${data.look.casaco} + ${data.look.tronco}`;
        atualizar(elTronco, troncoTexto);

        atualizar(elPernas, data.look.pernas);
        atualizar(elPes, data.look.pes);

        infoClimaTexto.innerText = `Look ideal para ${data.temperatura}°C (${data.categoria_clima})`;
        
        await carregarClima();

    } catch (err) {
        console.error(err);
        alert("Erro ao conectar.");
    } finally {
        btn.innerHTML = textoOriginal;
        btn.disabled = false;
    }
});

window.onload = carregarClima;