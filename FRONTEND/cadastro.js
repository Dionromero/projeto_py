const tagInput = document.getElementById('tagInput');
const tagContainer = document.getElementById('tagContainer');

// Array para armazenar as tags
let listaDeTags = [];

// Foca no input ao clicar na área cinza
tagContainer.addEventListener('click', (e) => {
    if(e.target === tagContainer) {
        tagInput.focus();
    }
});

// Adiciona tag ao apertar Enter ou Vírgula
tagInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ',') {
        e.preventDefault();
        processarInput();
    }
    // Apaga a última tag se apertar Backspace com input vazio
    if (e.key === 'Backspace' && tagInput.value === '' && listaDeTags.length > 0) {
        removerTag(listaDeTags.length - 1);
    }
});

// Adiciona tag ao clicar fora do campo (Blur)
tagInput.addEventListener('blur', function() {
    processarInput();
});

function processarInput() {
    const valor = tagInput.value.trim().replace(',', ''); // Remove vírgula se tiver
    if (valor !== "" && !listaDeTags.includes(valor)) {
        listaDeTags.push(valor);
        renderizarTags();
    }
    tagInput.value = "";
}

function removerTag(indice) {
    listaDeTags.splice(indice, 1);
    renderizarTags();
}

function renderizarTags() {
    // Limpa apenas as tags visuais (mantém o input)
    const tagsAtuais = document.querySelectorAll('.tag-badge');
    tagsAtuais.forEach(el => el.remove());

    // Recria as tags
    listaDeTags.forEach((tagTexto, index) => {
        const div = document.createElement('div');
        // Estilo da Tag (Azul bonito)
        div.className = 'tag-badge bg-blue-100 text-blue-700 px-3 py-1 rounded-lg text-sm font-bold flex items-center gap-2 select-none';
        
        // Texto da Tag
        const spanTexto = document.createElement('span');
        spanTexto.innerText = tagTexto;
        
        // Botão de Fechar (X)
        const spanClose = document.createElement('span');
        spanClose.innerHTML = '&times;';
        spanClose.className = 'cursor-pointer hover:text-blue-900 text-lg leading-none';
        
        // Evento de remover (Mais seguro que onclick no HTML)
        spanClose.onclick = function() {
            removerTag(index);
        };

        div.appendChild(spanTexto);
        div.appendChild(spanClose);
        
        // Insere ANTES do input
        tagContainer.insertBefore(div, tagInput);
    });
}

// ENVIO PARA O BACKEND
async function enviarCadastro() {
    // Garante que a tag pendente seja adicionada antes de enviar
    processarInput();

    const nome = document.getElementById('nomeRoupa').value;
    const imagem = document.getElementById('imgRoupa').value;

    if (!nome || !imagem) {
        alert("Por favor, preencha o nome e a URL da imagem!");
        return;
    }

    const dadosParaEnviar = {
        name: nome,
        image_path: imagem,
        tags: listaDeTags 
    };

    console.log("Enviando:", dadosParaEnviar);

    try {
        const resposta = await fetch('http://localhost:5000/api/clothes', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dadosParaEnviar)
        });

        if (resposta.ok) {
            alert("Roupa salva com sucesso!");
            // Limpar tudo
            document.getElementById('nomeRoupa').value = "";
            document.getElementById('imgRoupa').value = "";
            listaDeTags = [];
            renderizarTags();
        } else {
            alert("Erro ao cadastrar roupa.");
        }
    } catch (erro) {
        console.error("Erro:", erro);
        alert("Erro de conexão com o servidor.");
    }
}