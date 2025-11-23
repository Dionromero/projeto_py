// cadastro.js

const tagInput = document.getElementById('tagInput');
const tagContainer = document.getElementById('tagContainer');

// Array para armazenar as tags
let listaDeTags = [];

// Foca no input ao clicar na div container
tagContainer.addEventListener('click', () => {
    tagInput.focus();
});

// Evento: Quando o usuário aperta uma tecla no input de tag
tagInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        e.preventDefault(); 
        const valor = tagInput.value.trim();
        
        // Validação: Não vazio e não duplicado
        if (valor !== "" && !listaDeTags.includes(valor)) {
            adicionarTag(valor);
        }
        tagInput.value = ""; 
    }
});

function adicionarTag(texto) {
    listaDeTags.push(texto);
    renderizarTags();
}

function removerTag(indice) {
    listaDeTags.splice(indice, 1); 
    renderizarTags();
}

function renderizarTags() {
    // Remove tags visuais antigas (exceto o input)
    document.querySelectorAll('.tag').forEach(el => el.remove());

    // Recria as tags baseadas no Array
    listaDeTags.slice().reverse().forEach((tagTexto, index) => {
        const realIndex = listaDeTags.length - 1 - index;

        const div = document.createElement('div');
        div.className = 'tag';
        div.innerHTML = `
            ${tagTexto}
            <span onclick="removerTag(${realIndex})">&times;</span>
        `;
        
        // Insere a tag ANTES do input
        tagContainer.insertBefore(div, tagInput);
    });
}

// ENVIO PARA O BACKEND
async function enviarCadastro() {
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

    console.log("Enviando JSON:", dadosParaEnviar);

    try {
        const resposta = await fetch('http://localhost:5000/api/clothes', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dadosParaEnviar)
        });

        if (resposta.ok) {
            alert("✅ Roupa cadastrada com sucesso!");
            // Limpar formulário
            document.getElementById('nomeRoupa').value = "";
            document.getElementById('imgRoupa').value = "";
            listaDeTags = [];
            renderizarTags();
        } else {
            alert("❌ Erro no servidor ao salvar.");
        }
    } catch (error) {
        console.error("Erro:", error);
        alert("❌ Falha ao conectar com o Backend.");
    }
}